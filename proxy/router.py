"""HTTP proxy router - forwards requests to llama.cpp servers.

Catch-all route for /v1/server/{server_id}/* that rewrites the path
and forwards to the appropriate local llama.cpp server instance.

Tracks server activity: increments use_count on request start, decrements
when the response fully completes. This allows the cache eviction timer to
fire based on actual request activity rather than external release calls.

For chat completion requests with a session_id, this router pins each
session to a llama.cpp slot via a per-server LRU map, eagerly injects
``id_slot`` / ``cache_prompt`` / ``n_cache_reuse`` into the upstream body,
and persists slot KV state to disk on LRU eviction so concurrent sessions
sharing the same server share KV cache reuse benefits without colliding
on slot 0.
"""

import asyncio
import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from config import PROXY_TIMEOUT, SLOT_SAVE_DIR
from middleware.prometheus_metrics import (
    prompt_body_bytes,
    prompt_fingerprint_total,
    prompt_first_divergence_byte,
    slot_lru_size,
    slot_resolutions_total,
    slot_restore_duration_seconds,
    slot_restore_total,
    slot_save_duration_seconds,
    slot_save_total,
)
from utils.logging import llmmllogger, _session_id_ctx

logger = llmmllogger.bind(component="proxy_router")

router = APIRouter()

# ---------------------------------------------------------------------------
# Slot pinning state
# ---------------------------------------------------------------------------

# server_id -> number of slots (discovered from upstream /slots endpoint or
# from --parallel)
_num_slots_cache: Dict[str, int] = {}
# session_id -> last activity timestamp (for session-aware slot cleanup)
_session_activity: Dict[str, float] = {}


class SlotLRU:
    """LRU map from session_id to slot_id for a single upstream server.

    Slots are a fixed pool sized to ``--parallel`` (capacity).  The first
    ``capacity`` distinct sessions claim slots 0..capacity-1.  After that,
    inserting a new session evicts the least-recently-used session, freeing
    its slot for reuse.

    ``touch(session_id)`` returns ``(slot_id, evicted)`` where ``evicted`` is
    ``None`` if no eviction happened, otherwise ``(evicted_session_id,
    evicted_slot_id)``.  The caller is responsible for persisting the
    evicted slot's KV state and restoring the new session's KV state.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        # session_id -> slot_id, ordered by recency (MRU at end)
        self._map: "OrderedDict[str, int]" = OrderedDict()
        # set of slot_ids currently in use
        self._used_slots: Set[int] = set()
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    async def touch(
        self, session_id: str
    ) -> Tuple[int, Optional[Tuple[str, int]]]:
        """Insert or refresh a session and return its slot.

        Returns ``(slot_id, evicted_session)`` where ``evicted_session`` is
        ``(session_id, slot_id)`` if a session was evicted to make room,
        otherwise ``None``.
        """
        async with self._lock:
            if session_id in self._map:
                slot_id = self._map.pop(session_id)
                self._map[session_id] = slot_id
                return slot_id, None

            evicted: Optional[Tuple[str, int]] = None
            if len(self._map) >= self._capacity:
                old_session, old_slot = self._map.popitem(last=False)
                self._used_slots.discard(old_slot)
                evicted = (old_session, old_slot)
                slot_id = old_slot
            else:
                # Allocate the lowest free slot id in [0, capacity)
                slot_id = -1
                for candidate in range(self._capacity):
                    if candidate not in self._used_slots:
                        slot_id = candidate
                        break
                if slot_id == -1:
                    # Shouldn't happen — _used_slots should always have a free
                    # slot when len(_map) < capacity — but be defensive.
                    slot_id = len(self._map) % self._capacity

            self._map[session_id] = slot_id
            self._used_slots.add(slot_id)
            return slot_id, evicted

    async def peek(self, session_id: str) -> Optional[int]:
        """Return the slot for ``session_id`` without touching recency."""
        async with self._lock:
            return self._map.get(session_id)


# server_id -> SlotLRU (different llama.cpp instances have independent slot pools)
_slot_lrus: Dict[str, SlotLRU] = {}
_slot_lrus_lock = asyncio.Lock()


# Reverse lookup used by the subprocess log drain in server_manager/base.py
# to attribute llama.cpp stdout/stderr lines (which only know slot IDs) to
# the session_id pinned to that slot.  Cheap, lock-free (SlotLRU._map mutates
# under its own asyncio.Lock; this is a best-effort read that may race with
# eviction — the worst case is logging a stale session_id, which is still
# more useful than no session_id).
def session_for_slot(server_id: str, slot_id: int) -> Optional[str]:
    """Return the session_id currently pinned to ``slot_id`` on ``server_id``.

    Returns ``None`` if no SlotLRU exists for the server yet (e.g. during
    startup before the first proxy request) or no session is pinned.
    """
    lru = _slot_lrus.get(server_id)
    if lru is None:
        return None
    for sid, sl in lru._map.items():
        if sl == slot_id:
            return sid
    return None


# Pattern that matches llama.cpp's "slot launch_slot_: id N | task M" /
# "slot print_timing: id N | task M | ..." / etc. format.  llama.cpp itself
# doesn't know about our session concept, but every slot-related line has the
# slot id sitting right after `id `.  Compiled once at import.
import re as _re
_LLAMACPP_SLOT_RE = _re.compile(r"\bslot\s+\w+:\s+id\s+(\d+)\b")


def slot_id_from_llamacpp_line(line: str) -> Optional[int]:
    """Best-effort extraction of llama.cpp's slot id from a log line.

    Returns ``None`` for lines that don't reference a slot.
    """
    m = _LLAMACPP_SLOT_RE.search(line)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


async def _get_slot_lru(server_id: str, target_host: str) -> SlotLRU:
    """Return the per-server SlotLRU, creating it lazily."""
    lru = _slot_lrus.get(server_id)
    if lru is not None:
        return lru
    async with _slot_lrus_lock:
        lru = _slot_lrus.get(server_id)
        if lru is not None:
            return lru
        num_slots = await _discover_num_slots(target_host, server_id)
        lru = SlotLRU(capacity=num_slots)
        _slot_lrus[server_id] = lru
        logger.info(
            "Created SlotLRU", server_id=server_id, capacity=num_slots
        )
        return lru


# ---------------------------------------------------------------------------
# Lazy startup scan of SLOT_SAVE_DIR — what session files exist on disk
# ---------------------------------------------------------------------------

_known_session_files: Set[str] = set()
_known_files_loaded: bool = False
_known_files_lock = asyncio.Lock()


def _stable_model_key(server_id: str) -> str:
    """Resolve a server_id to a *stable* identifier that survives runner restarts.

    ``server_id`` is an ephemeral UUID assigned to each llama.cpp subprocess
    — it changes on every runner pod restart even when the same model is
    loaded.  Using it in slot-save filenames causes saved KV state to be
    orphaned across restarts.

    Instead, look up the ``model_id`` from ``server_cache``.  Model IDs are
    deterministic (driven by ``.models.yaml``) so the resulting filename
    survives restarts.  We sanitize the model_id to be filesystem-safe
    (no slashes, no colons).

    On any lookup failure (cache not populated yet, server_id unknown)
    fall back to ``server_id`` — restore won't hit but at least no crash.
    """
    if not server_id:
        return ""
    try:
        from cache import server_cache  # local import to avoid cycles at module-load
        entry = server_cache._servers.get(server_id)
        if entry is not None:
            model_id = getattr(entry, "model_id", None) or server_id
            # Filesystem-safe: replace path separators and ":" (HF-style).
            return str(model_id).replace("/", "_").replace(":", "_").replace("\\", "_")
    except Exception:
        pass
    return server_id


def _session_file_key(session_id: str, server_id: str) -> str:
    """Identifier we record in the in-memory `known files` set.

    Mirrors the on-disk filename (basename without extension) produced by
    ``_slot_file_path`` so we can quickly answer "does a save file exist for
    this (session, model)?" without hitting the filesystem.
    """
    model_key = _stable_model_key(server_id)
    if model_key:
        return f"slot_{session_id}_{model_key}"
    return f"slot_{session_id}"


async def _ensure_known_files_loaded() -> None:
    """One-time inventory of SLOT_SAVE_DIR.

    Populates ``_known_session_files`` with the basename (no extension) of
    every ``*.bin`` file in SLOT_SAVE_DIR so subsequent requests can answer
    "does a save exist for this session?" without a stat call.
    """
    global _known_files_loaded
    if _known_files_loaded or not SLOT_SAVE_DIR:
        return
    async with _known_files_lock:
        if _known_files_loaded:
            return
        try:
            if os.path.isdir(SLOT_SAVE_DIR):
                for name in os.listdir(SLOT_SAVE_DIR):
                    if name.endswith(".bin"):
                        _known_session_files.add(name[:-4])
            logger.info(
                "Inventoried SLOT_SAVE_DIR",
                slot_save_dir=SLOT_SAVE_DIR,
                file_count=len(_known_session_files),
            )
        except OSError as e:
            logger.warning(
                "Failed to inventory SLOT_SAVE_DIR",
                slot_save_dir=SLOT_SAVE_DIR,
                error=str(e),
            )
        _known_files_loaded = True


def _record_saved_file(session_id: str, server_id: str) -> None:
    """Refresh the in-memory set after we write a new save file."""
    _known_session_files.add(_session_file_key(session_id, server_id))


def _save_file_exists(session_id: str, server_id: str) -> bool:
    """Cheap check — is there a known save file for this session?"""
    return _session_file_key(session_id, server_id) in _known_session_files


# ---------------------------------------------------------------------------
# Prompt-divergence diagnostic (temporary).  Logs at which byte offset the
# request body first differs from the previous request body for the same
# session, so we can pin down why llama.cpp's prefix cache stops matching
# past a certain depth.  Cheap enough to leave on; remove once the
# divergence point is identified and fixed.
# ---------------------------------------------------------------------------

import hashlib as _hashlib  # local alias — hashlib is also imported lazily below

# session_id -> list of (offset, md5_short) snapshots from the previous request.
# Bounded via FIFO eviction so a long-running runner with many distinct
# session_ids doesn't accumulate unbounded entries (each entry is small —
# ~13 tuples of (int, 8-char hex) ≈ 600 B — but at 10k sessions that's
# 6 MiB of orphaned diagnostic state).
from collections import OrderedDict as _OrderedDict
_SESSION_PROMPT_HASHES_MAX = 1024
_session_prompt_hashes: "_OrderedDict[str, List[Tuple[int, str]]]" = _OrderedDict()

# Byte offsets at which we hash the prompt body.  Spans the range where we
# observed cache divergence (between ~16K and ~24K llama.cpp checkpoints).
_PROMPT_HASH_OFFSETS: Tuple[int, ...] = (
    1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576,
    32768, 49152, 65536, 98304, 131072,
)


def _hash_prefix(data: bytes, offset: int) -> str:
    """Short MD5 (first 8 hex chars) of ``data[:offset]``.

    Returns an empty string if the body is shorter than ``offset``.
    """
    if len(data) < offset:
        return ""
    return _hashlib.md5(data[:offset]).hexdigest()[:8]


def _log_prompt_divergence(session_id: str, body: bytes) -> None:
    """Compare body prefix-hashes to the prior request for this session.

    Logs one structured INFO line per request with: body length, all
    prefix hashes, and (if applicable) the first byte offset at which the
    hash differs from the previous request.
    """
    if not session_id or not body:
        return
    new_hashes: List[Tuple[int, str]] = [
        (off, _hash_prefix(body, off)) for off in _PROMPT_HASH_OFFSETS
    ]
    prev = _session_prompt_hashes.get(session_id)
    first_div: Optional[int] = None
    if prev:
        for (off_new, h_new), (off_prev, h_prev) in zip(new_hashes, prev):
            # Both must have data at this offset to be comparable.
            if not h_new or not h_prev:
                continue
            if h_new != h_prev:
                first_div = off_new
                break
    # FIFO-bounded write: refresh recency, then drop oldest if over cap.
    if session_id in _session_prompt_hashes:
        _session_prompt_hashes.move_to_end(session_id)
    _session_prompt_hashes[session_id] = new_hashes
    while len(_session_prompt_hashes) > _SESSION_PROMPT_HASHES_MAX:
        _session_prompt_hashes.popitem(last=False)

    # Build a compact representation: "1024:abcd1234,2048:..."
    hash_str = ",".join(f"{off}:{h or '-'}" for off, h in new_hashes)
    prompt_body_bytes.observe(len(body))
    if prev is None:
        prompt_fingerprint_total.labels(kind="first").inc()
        logger.info(
            "Prompt fingerprint (first turn)",
            session_id=session_id,
            body_len=len(body),
            hashes=hash_str,
        )
    elif first_div is None:
        prompt_fingerprint_total.labels(kind="stable").inc()
        logger.info(
            "Prompt fingerprint (prefix STABLE across all checkpoints)",
            session_id=session_id,
            body_len=len(body),
            hashes=hash_str,
        )
    else:
        prompt_fingerprint_total.labels(kind="diverged").inc()
        prompt_first_divergence_byte.observe(first_div)
        logger.info(
            "Prompt fingerprint (DIVERGED from previous turn)",
            session_id=session_id,
            body_len=len(body),
            first_divergence_byte=first_div,
            hashes=hash_str,
            prev_hashes=",".join(f"{off}:{h or '-'}" for off, h in prev),
        )


# ---------------------------------------------------------------------------
# Per-upstream-server httpx.AsyncClient pool
# ---------------------------------------------------------------------------

_http_clients: Dict[str, httpx.AsyncClient] = {}
_http_clients_lock = asyncio.Lock()


async def _get_http_client(target_host: str, parallel: int) -> httpx.AsyncClient:
    """Return (and lazily create) the shared httpx client for an upstream."""
    client = _http_clients.get(target_host)
    if client is not None and not client.is_closed:
        return client
    async with _http_clients_lock:
        client = _http_clients.get(target_host)
        if client is not None and not client.is_closed:
            return client
        limits = httpx.Limits(
            max_connections=max(parallel * 4, 4),
            max_keepalive_connections=max(parallel, 2),
        )
        client = httpx.AsyncClient(timeout=PROXY_TIMEOUT, limits=limits)
        _http_clients[target_host] = client
        logger.info(
            "Created upstream httpx client",
            target_host=target_host,
            max_connections=limits.max_connections,
            max_keepalive=limits.max_keepalive_connections,
        )
        return client


async def aclose_all_clients() -> None:
    """Close all pooled upstream clients.  Call from app shutdown."""
    async with _http_clients_lock:
        clients = list(_http_clients.values())
        _http_clients.clear()
    for c in clients:
        try:
            await c.aclose()
        except Exception as e:
            logger.debug("Error closing upstream client", error=str(e))


async def save_all_active_slots() -> int:
    """Save every slot currently pinned in a SlotLRU to disk.

    Called from app shutdown (SIGTERM path).  Iterates every server's
    SlotLRU, and for each (session_id, slot_id) pair issues one
    /slots/{id}?action=save call to the upstream llama.cpp server.
    Best-effort: individual save failures are logged but don't stop the
    sweep.

    Returns the number of slots successfully persisted.
    """
    if not SLOT_SAVE_DIR:
        return 0
    from app import server_cache  # late import — avoid cycle

    # Snapshot the LRUs so we don't iterate while concurrent requests
    # mutate them.  After this point, even if a request comes in and
    # rewrites the slot's KV, we save what we have right now.
    snapshots: List[Tuple[str, str, str, int]] = []
    for server_id, lru in list(_slot_lrus.items()):
        entry = server_cache.get(server_id)
        if entry is None:
            continue
        target_host = f"http://127.0.0.1:{entry.port}"
        # SlotLRU._map is session_id → slot_id.  Take a snapshot copy.
        for session_id, slot_id in list(lru._map.items()):
            snapshots.append((server_id, target_host, session_id, slot_id))

    if not snapshots:
        logger.info("save_all_active_slots: no active slots to persist")
        return 0

    logger.info(
        "save_all_active_slots: persisting active slots",
        extra={"count": len(snapshots)},
    )

    saved = 0
    for server_id, target_host, session_id, slot_id in snapshots:
        try:
            slot_file = _slot_file_path(session_id, server_id)
            # Use a fresh short-lived client — the pooled clients may
            # already be closing as part of shutdown.
            async with httpx.AsyncClient(timeout=30.0) as client:
                ok = await _save_slot(
                    target_host,
                    slot_id,
                    slot_file,
                    session_id=session_id,
                    client=client,
                )
                if ok:
                    _record_saved_file(session_id, server_id)
                    saved += 1
        except Exception as e:
            logger.warning(
                "save_all_active_slots: save raised",
                session_id=session_id,
                slot_id=slot_id,
                server_id=server_id,
                error=str(e),
            )
    return saved


# ---------------------------------------------------------------------------
# Session activity tracking (consumed by app.py's slot cleanup task)
# ---------------------------------------------------------------------------


def touch_session_activity(session_id: str) -> None:
    """Record that a session had a turn, updating the activity timestamp."""
    _session_activity[session_id] = time.time()


def get_session_activity(session_id: str) -> Optional[float]:
    """Get the last activity timestamp for a session, or None if unknown."""
    return _session_activity.get(session_id)


# ---------------------------------------------------------------------------
# SSE → non-streaming JSON conversion (unchanged)
# ---------------------------------------------------------------------------


def _sse_to_nonstreaming(chunks: bytes, model: str) -> Optional[Dict[str, Any]]:
    """Convert buffered SSE (stream=true) response to non-streaming JSON.

    Parses ``data: {...}`` lines, concatenates content, and builds a
    standard OpenAI non-streaming chat completion response.
    """
    text_parts: List[str] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None

    for line in chunks.decode(errors="replace").split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = obj.get("choices") or []
        for c in choices:
            delta = c.get("delta") or {}
            role = delta.get("role")
            if role and not text_parts:
                continue
            content = delta.get("content", "")
            if content:
                text_parts.append(content)
            fr = c.get("finish_reason")
            if fr:
                finish_reason = fr
        if usage is None and "usage" in obj:
            usage = obj["usage"]

    if not text_parts and finish_reason is None:
        return None

    result: Dict[str, Any] = {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(text_parts) if text_parts else "",
                },
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if usage:
        result["usage"] = usage
    return result


_FILENAME_UNSAFE_RE = _re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_for_filename(s: str) -> str:
    """Replace filesystem-unsafe characters (notably ``:`` from the
    ``pck:`` session-id prefix) with ``_``.

    llama.cpp's ``/slots/{id}?action=save`` endpoint validates the
    ``filename`` parameter and returns HTTP 400 if it contains characters
    like ``:`` or ``/``.  The mapping is one-way + lossy in principle,
    but session ids only contain hex + dashes plus the optional ``pck:``
    prefix, so the collision risk is effectively zero.
    """
    return _FILENAME_UNSAFE_RE.sub("_", s)


def _slot_file_path(session_id: str, server_id: str = "") -> str:
    """Build the absolute path for a session's slot cache file.

    Uses the *model_id* (resolved via ``_stable_model_key``) rather than
    the ephemeral ``server_id`` so the file survives runner restarts.
    Different models still get different files (preventing cross-model
    restore failures from incompatible tensor shapes).

    The session id is filename-sanitised before being included in the
    path — llama.cpp rejects names with ``:`` (HTTP 400 from
    ``/slots/{id}?action=save``), which broke every save attempt for
    ``pck:*`` openclaw sessions.
    """
    safe_sid = _sanitize_for_filename(session_id)
    model_key = _stable_model_key(server_id)
    if model_key:
        return f"{SLOT_SAVE_DIR}/slot_{safe_sid}_{model_key}.bin"
    return f"{SLOT_SAVE_DIR}/slot_{safe_sid}.bin"


# Legacy hash-based resolver kept for tests that still import it.
def _resolve_slot_id(session_id: str, server_id: str, num_slots: int) -> int:
    """Deprecated: use SlotLRU. Retained only for backward-compat with tests."""
    import hashlib

    return int(hashlib.md5(session_id.encode()).hexdigest(), 16) % max(num_slots, 1)


# ---------------------------------------------------------------------------
# Slot save / restore (used by save-on-evict and the legacy non-streaming path)
# ---------------------------------------------------------------------------


async def _restore_slot(
    target_host: str,
    slot_id: int,
    slot_file: str,
    session_id: str = "",
    client: Optional[httpx.AsyncClient] = None,
) -> bool:
    """Restore a KV cache slot from disk before a chat completion.

    Returns True if the slot was successfully restored, False otherwise.
    Logs one INFO line per call with session_id, slot_id, action, and result.
    """
    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)
        own_client = True
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{target_host}/slots/{slot_id}?action=restore",
            json={"filename": os.path.basename(slot_file)},
        )
        try:
            body = resp.json()
        except Exception:
            body = {}
        n_restored = body.get("n_restored") if isinstance(body, dict) else None
        ok = resp.status_code == 200
        slot_restore_total.labels(
            slot_id=str(slot_id),
            outcome="success" if ok else "failure",
        ).inc()
        slot_restore_duration_seconds.observe(time.monotonic() - start)
        logger.info(
            "Slot restore",
            session_id=session_id,
            slot_id=slot_id,
            action="restore",
            slot_file=os.path.basename(slot_file),
            status=resp.status_code,
            n_restored=n_restored,
        )
        return ok
    except Exception as e:
        slot_restore_total.labels(slot_id=str(slot_id), outcome="failure").inc()
        slot_restore_duration_seconds.observe(time.monotonic() - start)
        logger.warning(
            "Slot restore failed",
            session_id=session_id,
            slot_id=slot_id,
            action="restore",
            slot_file=os.path.basename(slot_file),
            error=str(e),
        )
        return False
    finally:
        if own_client:
            await client.aclose()


async def _save_slot(
    target_host: str,
    slot_id: int,
    slot_file: str,
    session_id: str = "",
    client: Optional[httpx.AsyncClient] = None,
) -> bool:
    """Save the KV cache slot to disk.

    Returns True on success.  Logs one INFO line per call.
    """
    own_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)
        own_client = True
    start = time.monotonic()
    try:
        resp = await client.post(
            f"{target_host}/slots/{slot_id}?action=save",
            json={"filename": os.path.basename(slot_file)},
        )
        try:
            body = resp.json()
        except Exception:
            body = {}
        n_saved = body.get("n_saved") if isinstance(body, dict) else None
        ok = resp.status_code == 200
        slot_save_total.labels(
            slot_id=str(slot_id),
            outcome="success" if ok else "failure",
        ).inc()
        slot_save_duration_seconds.observe(time.monotonic() - start)
        logger.info(
            "Slot save",
            session_id=session_id,
            slot_id=slot_id,
            action="save",
            slot_file=os.path.basename(slot_file),
            status=resp.status_code,
            n_saved=n_saved,
        )
        return ok
    except Exception as e:
        slot_save_total.labels(slot_id=str(slot_id), outcome="failure").inc()
        slot_save_duration_seconds.observe(time.monotonic() - start)
        logger.warning(
            "Slot save failed",
            session_id=session_id,
            slot_id=slot_id,
            action="save",
            slot_file=os.path.basename(slot_file),
            error=str(e),
        )
        return False
    finally:
        if own_client:
            await client.aclose()


async def _discover_num_slots(target_host: str, server_id: str) -> int:
    """Query upstream /slots to discover the number of available slots."""
    if server_id in _num_slots_cache:
        return _num_slots_cache[server_id]
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{target_host}/slots")
            if resp.status_code == 200:
                slots = resp.json()
                num = len(slots) if slots else 1
                _num_slots_cache[server_id] = num
                logger.info("Discovered slots", server_id=server_id, num_slots=num)
                # Bounce oldest entries to prevent memory leak from evicted servers
                if len(_num_slots_cache) > 100:
                    _num_slots_cache.pop(next(iter(_num_slots_cache)))
                return num
    except Exception as e:
        logger.warning(
            "Failed to discover slots, falling back to 1",
            server_id=server_id,
            error=str(e),
        )
    # Fallback
    _num_slots_cache[server_id] = 1
    return 1


# ---------------------------------------------------------------------------
# Body injection helper
# ---------------------------------------------------------------------------


def _inject_slot_body(
    body: bytes,
    slot_id: int,
    *,
    cache_prompt: bool = True,
    n_cache_reuse: int = 256,
) -> Tuple[bytes, bool]:
    """Inject id_slot / cache_prompt / n_cache_reuse into the JSON body.

    Returns ``(new_body, ok)``.  ``ok`` is False if the body wasn't JSON;
    the original body is returned unchanged in that case.

    Caller-provided values for any of the three keys are preserved.
    """
    if not body:
        return body, False
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return body, False
    if not isinstance(obj, dict):
        return body, False
    if "id_slot" not in obj:
        obj["id_slot"] = int(slot_id)
    if "cache_prompt" not in obj:
        obj["cache_prompt"] = cache_prompt
    if "n_cache_reuse" not in obj:
        obj["n_cache_reuse"] = n_cache_reuse
    return json.dumps(obj).encode("utf-8"), True


# ---------------------------------------------------------------------------
# Streaming proxy
# ---------------------------------------------------------------------------


async def _stream_upstream(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[bytes],
    server_id: str,
    target_host: str = "",
    slot_id: int = 0,
    slot_file: str = "",
    session_id: str = "",
):
    """Stream response from upstream using a shared httpx client.

    The client is owned by the per-server pool and is NOT closed here.
    Decrements the server's use_count when the stream fully drains or the
    downstream client disconnects.

    Persistence behaviour: when called with a non-empty ``slot_file`` and
    ``session_id``, the slot's KV state is saved to disk AFTER the stream
    drains successfully.  This is the "save after every turn" half of the
    aggressive-persistence design — combined with save-on-evict (in
    ``proxy_request``) and save-on-SIGTERM (in ``app.py`` lifespan), it
    ensures that the next request for the same session has a fresh disk
    snapshot to restore from, regardless of how the slot's in-memory KV
    was lost (LRU eviction, unified-KV pressure, pod restart, OOM kill).
    """
    try:
        response = await client.send(
            client.build_request(
                method=method,
                url=url,
                headers=headers,
                content=body if body else None,
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        logger.error("Upstream server %s error before response: %s", server_id, exc)
        from app import server_cache

        server_cache.decrement_use(server_id)
        raise

    response_headers = dict(response.headers)
    status_code = response.status_code
    clean_headers = {
        k: v
        for k, v in response_headers.items()
        if k.lower() not in ("transfer-encoding", "content-length")
    }

    # Only save on a 2xx outcome — a 4xx/5xx upstream response means the
    # slot's KV is in an undefined state and we'd be persisting garbage.
    should_save_after = (
        bool(slot_file)
        and bool(session_id)
        and bool(target_host)
        and 200 <= status_code < 300
    )

    async def upstream_iterator():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            try:
                await response.aclose()
            except Exception:
                pass
            from app import server_cache

            server_cache.decrement_use(server_id)

            # Save the slot's KV to disk now that the turn's tokens have
            # been written into it.  Best-effort: failure here doesn't
            # affect the response the client already received.
            if should_save_after:
                try:
                    saved = await _save_slot(
                        target_host,
                        slot_id,
                        slot_file,
                        session_id=session_id,
                        client=client,
                    )
                    if saved:
                        _record_saved_file(session_id, server_id)
                except Exception as exc:
                    logger.warning(
                        "Post-stream slot save raised",
                        session_id=session_id,
                        slot_id=slot_id,
                        error=str(exc),
                    )

    return StreamingResponse(
        content=upstream_iterator(),
        status_code=status_code,
        headers=clean_headers,
    )


# ---------------------------------------------------------------------------
# Slot save / restore proxy endpoints (unchanged public surface)
# ---------------------------------------------------------------------------


@router.post("/v1/server/{server_id}/slots/{slot_id}/save")
async def save_slot(request: Request, server_id: str, slot_id: int):
    """Save the KV cache slot to disk for session persistence.

    Proxies to llama.cpp's /slots/{slot_id}/save endpoint.
    The slot file is written to the directory configured by SLOT_SAVE_DIR.

    Returns the llama.cpp save response (slot index, path, success status).
    """
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    target_host = f"http://127.0.0.1:{entry.port}"
    upstream_url = f"{target_host}/slots/{slot_id}?action=save"

    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                upstream_url,
                content=body if body else b'{"filename":"default.bin"}',
                headers={"Content-Type": "application/json"},
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream server {server_id} is unreachable",
            )

    return {
        "status": "saved" if response.status_code == 200 else "failed",
        "slot_id": slot_id,
        "server_id": server_id,
        "upstream_status": response.status_code,
        "detail": response.text,
    }


@router.post("/v1/server/{server_id}/slots/{slot_id}/restore")
async def restore_slot(request: Request, server_id: str, slot_id: int):
    """Restore a KV cache slot from disk for session resumption.

    Proxies to llama.cpp's /slots/{slot_id}/restore endpoint.
    The slot file is read from the directory configured by SLOT_SAVE_DIR.

    Returns the llama.cpp restore response (slot index, path, success status).
    """
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    target_host = f"http://127.0.0.1:{entry.port}"
    upstream_url = f"{target_host}/slots/{slot_id}?action=restore"

    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                upstream_url,
                content=body if body else b'{"filename":"default.bin"}',
                headers={"Content-Type": "application/json"},
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream server {server_id} is unreachable",
            )

    return {
        "status": "restored" if response.status_code == 200 else "failed",
        "slot_id": slot_id,
        "server_id": server_id,
        "upstream_status": response.status_code,
        "detail": response.text,
    }


# ---------------------------------------------------------------------------
# Main catch-all proxy route
# ---------------------------------------------------------------------------


@router.api_route(
    "/v1/server/{server_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_request(request: Request, server_id: str, path: str):
    """Proxy a request to the target llama.cpp server.

    Path rewriting:
      /v1/server/{id}/v1/chat/completions  ->  http://127.0.0.1:{port}/v1/chat/completions
      /v1/server/{id}/health               ->  http://127.0.0.1:{port}/health

    SSE responses are streamed without buffering.

    For chat completions with a session_id, this route:
      1. Resolves (server, session) -> slot via a per-server LRU.
      2. On LRU eviction, saves the evicted slot's KV state to disk.
      3. If a saved file exists for the new session, restores it into the slot.
      4. Injects id_slot / cache_prompt / n_cache_reuse into the upstream body
         before forwarding.
    """
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    target_host = f"http://127.0.0.1:{entry.port}"

    # Lazy startup scan of SLOT_SAVE_DIR on first proxy call
    await _ensure_known_files_loaded()

    # Resolve session_id for slot persistence
    session_id = _session_id_ctx.get() or request.headers.get("x-session-id")
    is_chat_completion = "chat/completions" in path or path.endswith("completion")

    # Resolve the slot via the per-server LRU.  This works even when
    # SLOT_SAVE_DIR is unset — pinning + cache_prompt are still useful.
    slot_id: Optional[int] = None
    evicted_pair: Optional[Tuple[str, int]] = None
    slot_lru: Optional[SlotLRU] = None
    if session_id and is_chat_completion:
        slot_lru = await _get_slot_lru(server_id, target_host)
        slot_id, evicted_pair = await slot_lru.touch(session_id)
        touch_session_activity(session_id)
        slot_resolutions_total.labels(
            slot_id=str(slot_id),
            evicted="true" if evicted_pair is not None else "false",
        ).inc()
        slot_lru_size.labels(server_id=server_id).set(len(slot_lru._map))
        logger.info(
            "Resolved slot via LRU",
            session_id=session_id,
            server_id=server_id,
            slot_id=slot_id,
            evicted=evicted_pair[0] if evicted_pair else None,
        )

    # Determine whether persistence is active
    persistence_on = bool(SLOT_SAVE_DIR) and session_id and is_chat_completion
    num_slots = _num_slots_cache.get(server_id, 1)
    pool_client = await _get_http_client(target_host, num_slots)

    # Step 3a: Save the evicted session's slot, then restore the new
    # session's file (if any).  Both are awaited so the upstream is in
    # the right state before we forward.
    if persistence_on and evicted_pair is not None:
        old_session, old_slot = evicted_pair
        old_file = _slot_file_path(old_session, server_id)
        if await _save_slot(
            target_host,
            old_slot,
            old_file,
            session_id=old_session,
            client=pool_client,
        ):
            _record_saved_file(old_session, server_id)

    if persistence_on and slot_id is not None and session_id:
        if _save_file_exists(session_id, server_id):
            new_file = _slot_file_path(session_id, server_id)
            await _restore_slot(
                target_host,
                slot_id,
                new_file,
                session_id=session_id,
                client=pool_client,
            )

    # Mark server as in-use during request
    server_cache.increment_use(server_id)
    is_streaming = False

    try:
        # Rewrite path: strip the /v1/server/{id} prefix
        upstream_url = f"{target_host}/{path}"

        # Read request body
        body = await request.body()

        # Diagnostic: log prefix-hash divergence vs prior turn for this session.
        # Body is hashed BEFORE our slot/cache injection so we measure only
        # api-side variation.
        if is_chat_completion and session_id and body:
            _log_prompt_divergence(session_id, body)

        # Step 2: eager body injection for chat completions with a pinned slot
        upstream_body = body
        if slot_id is not None and is_chat_completion and body:
            new_body, ok = _inject_slot_body(body, slot_id)
            if ok:
                upstream_body = new_body
            else:
                logger.warning(
                    "Could not parse request body as JSON — skipping slot/cache "
                    "injection",
                    session_id=session_id,
                    slot_id=slot_id,
                    server_id=server_id,
                )

        # Build headers (exclude hop-by-hop)
        hop_by_hop = {
            "host",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "upgrade",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "proxy-connection",
            "content-length",  # stripped — body may be modified
        }
        headers = dict(request.headers)
        for h in hop_by_hop:
            headers.pop(h, None)
        headers["host"] = f"127.0.0.1:{entry.port}"

        method = request.method

        # Decide streaming vs non-streaming based on the (possibly injected) body
        is_likely_sse = method == "POST" and is_chat_completion and bool(body)
        if is_likely_sse:
            try:
                bd = json.loads(upstream_body if upstream_body else body)
                is_likely_sse = bool(bd.get("stream", False))
            except Exception:
                is_likely_sse = True  # be safe — stream if unparseable

        logger.info(
            "Proxy path",
            server_id=server_id,
            path=path,
            is_streaming=is_likely_sse,
            slot_id=slot_id,
            session_id=session_id,
        )

        if is_likely_sse:
            is_streaming = True
            return await _stream_upstream(
                pool_client,
                method,
                upstream_url,
                headers,
                upstream_body,
                server_id,
                target_host=target_host,
                slot_id=slot_id or 0,
                slot_file=(
                    _slot_file_path(session_id, server_id)
                    if persistence_on and session_id
                    else ""
                ),
                session_id=session_id or "",
            )

        # Non-streaming path — use the pooled client
        try:
            async with pool_client.stream(
                method=method,
                url=upstream_url,
                headers=headers,
                content=upstream_body if upstream_body else None,
            ) as response:
                resp_content = b""
                async for chunk in response.aiter_bytes():
                    resp_content += chunk

                # Save-after-turn for the non-streaming path.  Mirrors the
                # post-stream save in ``_stream_upstream``.  Only on 2xx —
                # a 4xx/5xx upstream means the slot state is undefined.
                if (
                    persistence_on
                    and slot_id is not None
                    and session_id
                    and 200 <= response.status_code < 300
                ):
                    try:
                        new_file = _slot_file_path(session_id, server_id)
                        saved = await _save_slot(
                            target_host,
                            slot_id,
                            new_file,
                            session_id=session_id,
                            client=pool_client,
                        )
                        if saved:
                            _record_saved_file(session_id, server_id)
                    except Exception as exc:
                        logger.warning(
                            "Post-completion slot save raised (non-streaming)",
                            session_id=session_id,
                            slot_id=slot_id,
                            error=str(exc),
                        )

                return Response(
                    content=resp_content,
                    status_code=response.status_code,
                    headers={
                        k: v
                        for k, v in dict(response.headers).items()
                        if k.lower() not in ("transfer-encoding", "content-length")
                    },
                )
        except httpx.HTTPError:
            raise

    except httpx.RemoteProtocolError as exc:
        logger.error("Upstream server %s disconnected: %s", server_id, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Upstream server {server_id} disconnected unexpectedly: {exc}",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream server {server_id} on port {entry.port} is unreachable",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Upstream server {server_id} timed out",
        )
    finally:
        # Streaming requests decrement inside _stream_upstream's iterator.
        if not is_streaming:
            server_cache.decrement_use(server_id)
