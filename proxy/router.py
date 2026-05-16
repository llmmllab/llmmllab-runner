"""HTTP proxy router - forwards requests to llama.cpp servers.

Catch-all route for /v1/server/{server_id}/* that rewrites the path
and forwards to the appropriate local llama.cpp server instance.

Tracks server activity: increments use_count on request start, decrements
when the response fully completes. This allows the cache eviction timer to
fire based on actual request activity rather than external release calls.

For chat completion requests with a session_id, automatically restores the
KV cache slot before forwarding and saves it after the response drains,
enabling persistent conversation state across requests.
"""

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from config import PROXY_TIMEOUT, SLOT_SAVE_DIR
from utils.logging import llmmllogger, _session_id_ctx

logger = llmmllogger.bind(component="proxy_router")

router = APIRouter()

# server_id -> number of slots (discovered from upstream /slots endpoint)
_num_slots_cache: Dict[str, int] = {}
# session_id -> slot_id (discovered from first request response)
_session_slot_cache: Dict[str, int] = {}


def _sse_to_nonstreaming(
    chunks: bytes, model: str
) -> Optional[Dict[str, Any]]:
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


def _slot_file_path(session_id: str) -> str:
    """Build the absolute path for a session's slot cache file."""
    return f"{SLOT_SAVE_DIR}/slot_{session_id}.bin"


def _resolve_slot_id(session_id: str, server_id: str, num_slots: int) -> int:
    """Map a session_id to a slot index via hash.

    Ensures the same session always uses the same slot across requests.
    """
    return int(hashlib.md5(session_id.encode()).hexdigest(), 16) % num_slots


async def _restore_slot(target_host: str, slot_id: int, slot_file: str) -> bool:
    """Restore a KV cache slot from disk before a chat completion.

    Returns True if the slot was successfully restored, False otherwise.
    Silently skips if the slot file doesn't exist (first request for session).
    """
    if not os.path.exists(slot_file):
        logger.debug("Slot file does not exist, skipping restore", slot_file=slot_file)
        return False
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{target_host}/slots/{slot_id}/restore",
                json={"filename": os.path.basename(slot_file)},
            )
            logger.info(
                "Slot restore response",
                slot_file=slot_file,
                slot_id=slot_id,
                status=resp.status_code,
                body=resp.text[:200],
            )
            return resp.status_code == 200
    except Exception as e:
        logger.warning("Slot restore failed", slot_file=slot_file, slot_id=slot_id, error=str(e))
        return False


async def _save_slot(target_host: str, slot_id: int, slot_file: str) -> None:
    """Save the KV cache slot to disk after a chat completion."""
    logger.info("Attempting slot save", target_host=target_host, slot_id=slot_id, slot_file=slot_file)
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(
                f"{target_host}/slots/{slot_id}/save",
                json={"filename": os.path.basename(slot_file)},
            )
            logger.info(
                "Slot save response",
                slot_file=slot_file,
                slot_id=slot_id,
                status=resp.status_code,
                body=resp.text[:200],
            )
    except Exception as e:
        logger.warning("Slot save failed", slot_file=slot_file, slot_id=slot_id, error=str(e))


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
        logger.warning("Failed to discover slots, falling back to 1", server_id=server_id, error=str(e))
    # Fallback
    _num_slots_cache[server_id] = 1
    return 1


async def _find_used_slot(target_host: str, num_slots: int, before_tokens: Dict[int, int] | None = None) -> int | None:
    """Find which slot llama.cpp actually used by querying /slots.

    llama.cpp doesn't send x-slot-id in response headers, so we discover
    the used slot by comparing token counts before and after the request.
    If no baseline is provided, falls back to the slot that's processing.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{target_host}/slots")
            if resp.status_code == 200:
                slots = resp.json()
                if before_tokens is not None:
                    # Find the slot whose token count changed
                    for s in slots:
                        sid = s.get("id")
                        n = s.get("n_tokens", 0)
                        if sid is not None and n != before_tokens.get(sid, 0):
                            return sid
                # Fallback: slot that's actively processing
                for s in slots:
                    if s.get("is_processing", False):
                        return s.get("id")
    except Exception as e:
        logger.warning("Failed to find used slot", error=str(e))
    return None


async def _stream_upstream(
    client, method, url, headers, body, server_id,
    target_host: str = "",
    slot_id: int = 0,
    slot_file: str = "",
    session_id: str = "",
    before_tokens: Dict[int, int] | None = None,
):
    """Stream response from upstream, keeping client open for the entire duration.

    Returns a StreamingResponse that maintains the upstream connection
    until the downstream client is done consuming. Decrements the server's
    use count when the stream fully completes or is abandoned.

    When the downstream client disconnects mid-stream, ``aclose()`` is
    called on the upstream response which closes the TCP connection to
    llama.cpp, causing it to stop generating tokens for the abandoned slot.

    If slot_file is provided, saves the KV cache slot after the stream drains.
    """
    import asyncio

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
        # Upstream crashed, refused connection, or timed out before sending
        # any response headers. Decrement use_count here because the caller's
        # finally block skips it when is_streaming is True.
        logger.error("Upstream server %s error before response: %s", server_id, exc)
        await client.aclose()
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

    # Track which slot llama.cpp actually used, discovered on first chunk
    actual_slot_id = [None]

    async def upstream_iterator():
        try:
            async for chunk in response.aiter_bytes():
                # Discover the actual slot on the first chunk by diffing
                # slot token counts against the pre-request baseline.
                if before_tokens and actual_slot_id[0] is None:
                    actual_slot_id[0] = await _find_used_slot(
                        target_host,
                        _num_slots_cache.get(server_id, 1),
                        before_tokens,
                    )
                yield chunk
        finally:
            # Save KV cache slot BEFORE closing the upstream connection.
            # The slot still holds its KV cache data while the connection
            # is open.  Once aclose() fires, llama.cpp may free the slot
            # and the save would get nothing or a 404.  Use a short timeout
            # so a hung save doesn't stall the response.
            if slot_file and target_host:
                if actual_slot_id[0] is None:
                    actual_slot_id[0] = slot_id
                if session_id:
                    _session_slot_cache[session_id] = actual_slot_id[0]
                try:
                    await asyncio.wait_for(_save_slot(target_host, actual_slot_id[0], slot_file), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Slot save timed out (5s), skipping", slot_file=slot_file)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("Slot save failed in finally", error=str(e))

            # Close the upstream response (aborts connection if still open,
            # which signals llama.cpp to stop generating for this slot)
            await response.aclose()
            await client.aclose()

            from app import server_cache
            server_cache.decrement_use(server_id)

    return StreamingResponse(
        content=upstream_iterator(),
        status_code=status_code,
        headers=clean_headers,
    )


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
    upstream_url = f"{target_host}/slots/{slot_id}/save"

    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                upstream_url,
                content=body,
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
    upstream_url = f"{target_host}/slots/{slot_id}/restore"

    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                upstream_url,
                content=body,
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

    For chat completions with a session_id, restores the KV cache slot
    before forwarding and saves it after the response drains.
    """
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    target_host = f"http://127.0.0.1:{entry.port}"

    # Resolve session_id for slot persistence
    session_id = _session_id_ctx.get() or request.headers.get("x-session-id")
    is_chat_completion = "chat/completions" in path

    # Slot persistence state (only for chat completions with session_id)
    slot_id = 0
    slot_file = ""
    if SLOT_SAVE_DIR and session_id and is_chat_completion:
        num_slots = await _discover_num_slots(target_host, server_id)
        slot_id = _resolve_slot_id(session_id, server_id, num_slots)
        slot_file = _slot_file_path(session_id)
        logger.info(
            "Slot persistence enabled",
            session_id=session_id,
            slot_id=slot_id,
            slot_file=slot_file,
            slot_save_dir=SLOT_SAVE_DIR,
        )
    elif is_chat_completion:
        logger.info(
            "Slot persistence skipped",
            has_slot_dir=bool(SLOT_SAVE_DIR),
            has_session_id=bool(session_id),
            is_chat=is_chat_completion,
        )

    # Reject chat completion requests when all slots are busy to prevent
    # request queueing that causes cascading timeouts with --parallel 1.
    # Include a Retry-After header so callers can back off instead of
    # hammering the endpoint in a tight retry loop.
    # Only reject if the server is healthy (responding to /health) AND
    # all slots report as busy — a freshly restarted server may show stale
    # slot state, so we skip the check if /health fails.
    remaining = path
    if is_chat_completion:
        try:
            async with httpx.AsyncClient(timeout=2.0) as check_client:
                # Verify the server is actually responsive first
                health_resp = await check_client.get(f"{target_host}/health")
                if health_resp.status_code == 200:
                    slots_resp = await check_client.get(f"{target_host}/slots")
                    if slots_resp.status_code == 200:
                        slots = slots_resp.json()
                        # Only reject when there are slots AND all of them
                        # are actively processing.  An empty slots list
                        # (e.g., misconfigured server) should not be
                        # treated as "all busy" — let the upstream handle it.
                        if slots and all(
                            s.get("is_processing", False) for s in slots
                        ):
                            # Estimate remaining time from the busiest slot.
                            # Fall back to 30 s if we can't estimate.
                            retry_after = 30
                            for s in slots:
                                if not s.get("is_processing"):
                                    continue
                                next_token = s.get("next_token") or {}
                                n_remain = next_token.get("n_remain")
                                n_decoded = next_token.get("n_decoded", 0)
                                # n_ctx is the slot's context window;
                                # tokens left ≈ n_ctx - n_decoded when
                                # n_remain is unavailable or negative.
                                n_ctx = s.get("n_ctx", 0)
                                if n_remain is None or n_remain < 0:
                                    n_remain = max(n_ctx - n_decoded, 1)
                                # Try to get token speed from the slot.
                                # Fall back to 20 tokens/s if unavailable.
                                t_token_ms = s.get("t_token_ms")
                                if t_token_ms is None or t_token_ms <= 0:
                                    t_token_ms = 50  # 20 tok/s default
                                estimated_ms = n_remain * t_token_ms
                                # Add 20% safety margin, round up to seconds
                                slot_retry = int(estimated_ms * 1.2 / 1000) + 1
                                retry_after = max(retry_after, slot_retry)
                            logger.warning(
                                f"All slots busy, rejecting request "
                                f"(retry_after={retry_after}s)"
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="All inference slots are busy",
                                headers={"Retry-After": str(retry_after)},
                            )
        except HTTPException:
            raise
        except Exception:
            pass  # If /health or /slots check fails, proceed normally

    # Restore KV cache slot before chat completion
    if slot_file:
        # Use cached slot ID if we already know which slot llama.cpp assigned
        restore_slot_id = _session_slot_cache.get(session_id, slot_id)
        await _restore_slot(target_host, restore_slot_id, slot_file)

    # Mark server as in-use during request
    server_cache.increment_use(server_id)
    is_streaming = False
    slot_saved = False

    try:
        # Rewrite path: strip the /v1/server/{id} prefix
        upstream_url = f"{target_host}/{remaining}"

        # Read request body
        body = await request.body()

        # Inject id_slot only when we know the slot from a previous
        # request. This ensures llama.cpp uses the same slot we saved to.
        upstream_body = body
        if slot_file and is_chat_completion and body and session_id in _session_slot_cache:
            try:
                body_dict = json.loads(body)
                body_dict["id_slot"] = _session_slot_cache[session_id]
                upstream_body = json.dumps(body_dict)
            except Exception:
                pass

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
            "content-length",  # stripped — body may be modified (id_slot)
        }
        headers = dict(request.headers)
        for h in hop_by_hop:
            headers.pop(h, None)
        headers["host"] = f"127.0.0.1:{entry.port}"

        method = request.method
        client = httpx.AsyncClient(timeout=PROXY_TIMEOUT)

        # Check if this is likely an SSE request (POST to /v1/chat/completions)
        is_likely_sse = method == "POST" and is_chat_completion and body

        if is_likely_sse:
            try:
                body_dict = json.loads(body)
                is_likely_sse = body_dict.get("stream", False)
            except Exception:
                is_likely_sse = True  # be safe, stream if we can't parse

        logger.info(
            "Proxy path",
            server_id=server_id,
            path=path,
            is_streaming=is_likely_sse,
            has_slot_file=bool(slot_file),
        )

        if is_likely_sse:
            # Capture slot token baseline before the request so we can
            # diff after to find which slot llama.cpp actually used.
            before_tokens: Dict[int, int] = {}
            if slot_file and target_host:
                try:
                    async with httpx.AsyncClient(timeout=2.0) as bl_client:
                        bl_resp = await bl_client.get(f"{target_host}/slots")
                        if bl_resp.status_code == 200:
                            for s in bl_resp.json():
                                sid = s.get("id")
                                if sid is not None:
                                    before_tokens[sid] = s.get("n_tokens", 0)
                except Exception:
                    pass

            # Streaming: decrement and slot save happen in upstream_iterator's
            # finally block when the stream drains or client disconnects.
            is_streaming = True
            return await _stream_upstream(
                client,
                method,
                upstream_url,
                headers,
                upstream_body,
                server_id,
                target_host=target_host,
                slot_id=slot_id,
                slot_file=slot_file,
                session_id=session_id or "",
                before_tokens=before_tokens if before_tokens else None,
            )

        # Non-streaming path
        logger.info("Taking non-streaming path", server_id=server_id, has_slot_file=bool(slot_file))

        # If slot persistence is active, always stream upstream to keep the
        # connection open (and the slot assigned) until we can save.  Then
        # convert the SSE back to non-streaming JSON for the client.
        if slot_file and is_chat_completion:
            try:
                body_dict = json.loads(upstream_body)
                body_dict["stream"] = True
                stream_body = json.dumps(body_dict)
            except Exception:
                stream_body = upstream_body
            logger.info("Overriding upstream to stream=true for slot save", server_id=server_id)

            try:
                response = await client.send(
                    client.build_request(
                        method=method,
                        url=upstream_url,
                        headers=headers,
                        content=stream_body,
                    ),
                    stream=True,
                )
            except httpx.HTTPError as exc:
                logger.error("Upstream server %s error before response: %s", server_id, exc)
                await client.aclose()
                server_cache.decrement_use(server_id)
                raise

            resp_content = b""
            async for chunk in response.aiter_bytes():
                resp_content += chunk

            # Save slot BEFORE closing the upstream connection — the slot
            # is still assigned while the connection is open.
            num = _num_slots_cache.get(server_id, 1)
            actual_slot = await _find_used_slot(target_host, num)
            if actual_slot is None:
                actual_slot = slot_id
            if session_id:
                _session_slot_cache[session_id] = actual_slot
            logger.info("Saving slot (non-streaming path)", slot_file=slot_file, slot_id=actual_slot)
            await _save_slot(target_host, actual_slot, slot_file)
            slot_saved = True

            await response.aclose()
            await client.aclose()
            server_cache.decrement_use(server_id)

            # Convert SSE back to non-streaming JSON
            model_name = ""
            try:
                body_dict = json.loads(body)
                model_name = body_dict.get("model", "")
            except Exception:
                pass
            result = _sse_to_nonstreaming(resp_content, model_name)
            if result:
                return Response(
                    content=json.dumps(result),
                    status_code=200,
                    media_type="application/json",
                )

            # Fallback: return raw upstream response if conversion fails
            return Response(
                content=resp_content,
                status_code=response.status_code,
                headers={
                    k: v
                    for k, v in dict(response.headers).items()
                    if k.lower() not in ("transfer-encoding", "content-length")
                },
            )

        # No slot persistence — straightforward non-streaming proxy
        async with client:
            async with client.stream(
                method=method,
                url=upstream_url,
                headers=headers,
                content=upstream_body if upstream_body else None,
            ) as response:
                resp_content = b""
                async for chunk in response.aiter_bytes():
                    resp_content += chunk

                return Response(
                    content=resp_content,
                    status_code=response.status_code,
                    headers={
                        k: v
                        for k, v in dict(response.headers).items()
                        if k.lower() not in ("transfer-encoding", "content-length")
                    },
                )

    except httpx.RemoteProtocolError as exc:
        # Upstream llama.cpp crashed or disconnected mid-request
        # (e.g., OOM, segfault, or killed by the kernel).
        # Return 503 so the caller can retry on a different server.
        logger.error(
            "Upstream server %s disconnected: %s", server_id, exc
        )
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
        # Decrement for non-streaming requests and errors.
        # Streaming requests decrement inside _stream_upstream's iterator
        # finally block when the stream fully drains or client disconnects.
        if not is_streaming and not slot_saved:
            # Save slot on error path for non-streaming chat completions
            if slot_file:
                await _save_slot(target_host, slot_id, slot_file)
            server_cache.decrement_use(server_id)
