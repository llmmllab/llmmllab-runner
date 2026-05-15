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

import hashlib
import json
import os
from typing import Dict

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from config import PROXY_TIMEOUT, SLOT_SAVE_DIR
from utils.logging import llmmllogger, _session_id_ctx

logger = llmmllogger.bind(component="proxy_router")

router = APIRouter()

# server_id -> number of slots (discovered from upstream /slots endpoint)
_num_slots_cache: Dict[str, int] = {}


def _slot_file_path(session_id: str) -> str:
    """Build the absolute path for a session's slot cache file."""
    return f"{SLOT_SAVE_DIR}/slot_{session_id}.bin"


def _resolve_slot_id(session_id: str, server_id: str, num_slots: int) -> int:
    """Map a session_id to a slot index via hash.

    Ensures the same session always uses the same slot across requests.
    """
    return int(hashlib.md5(session_id.encode()).hexdigest(), 16) % num_slots


async def _restore_slot(target_host: str, slot_id: int, slot_file: str) -> None:
    """Restore a KV cache slot from disk before a chat completion.

    Silently skips if the slot file doesn't exist (first request for session).
    """
    if not os.path.exists(slot_file):
        logger.debug("Slot file does not exist, skipping restore", slot_file=slot_file)
        return
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
    except Exception as e:
        logger.warning("Slot restore failed", slot_file=slot_file, slot_id=slot_id, error=str(e))


async def _save_slot(target_host: str, slot_id: int, slot_file: str) -> None:
    """Save the KV cache slot to disk after a chat completion."""
    logger.info("Attempting slot save", target_host=target_host, slot_id=slot_id, slot_file=slot_file)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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


async def _stream_upstream(
    client, method, url, headers, body, server_id,
    target_host: str = "",
    slot_id: int = 0,
    slot_file: str = "",
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

    async def upstream_iterator():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            # Close the upstream response (aborts connection if still open,
            # which signals llama.cpp to stop generating for this slot)
            await response.aclose()
            await client.aclose()
            # Save KV cache slot and decrement use count in a background task
            # to avoid being cancelled by the event loop when the client
            # disconnects.  The finally block runs inside a CancelledError
            # context, so any await can be immediately re-cancelled.
            if slot_file and target_host:
                asyncio.create_task(_safe_save_and_decrement(
                    target_host, slot_id, slot_file, server_id
                ))
            else:
                asyncio.create_task(_safe_decrement(server_id))

    return StreamingResponse(
        content=upstream_iterator(),
        status_code=status_code,
        headers=clean_headers,
    )


async def _safe_save_and_decrement(
    target_host: str, slot_id: int, slot_file: str, server_id: str
) -> None:
    """Save slot and decrement use count, immune to task cancellation."""
    try:
        await _save_slot(target_host, slot_id, slot_file)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("Background slot save failed", error=str(e))
    finally:
        from app import server_cache
        server_cache.decrement_use(server_id)


async def _safe_decrement(server_id: str) -> None:
    """Decrement use count in a background task."""
    from app import server_cache
    server_cache.decrement_use(server_id)


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
        await _restore_slot(target_host, slot_id, slot_file)

    # Mark server as in-use during request
    server_cache.increment_use(server_id)
    is_streaming = False
    slot_saved = False

    try:
        # Rewrite path: strip the /v1/server/{id} prefix
        upstream_url = f"{target_host}/{remaining}"

        # Read request body
        body = await request.body()

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
            # Streaming: decrement and slot save happen in upstream_iterator's
            # finally block when the stream drains or client disconnects.
            is_streaming = True
            return await _stream_upstream(
                client,
                method,
                upstream_url,
                headers,
                body,
                server_id,
                target_host=target_host,
                slot_id=slot_id,
                slot_file=slot_file,
            )

        # Non-streaming: buffer entire response
        logger.info("Taking non-streaming path", server_id=server_id, has_slot_file=bool(slot_file))
        async with client:
            async with client.stream(
                method=method,
                url=upstream_url,
                headers=headers,
                content=body if body else None,
            ) as response:
                content = b""
                async for chunk in response.aiter_bytes():
                    content += chunk

                # Save KV cache slot after non-streaming response
                if slot_file:
                    logger.info("Saving slot (non-streaming path)", slot_file=slot_file)
                    await _save_slot(target_host, slot_id, slot_file)
                    slot_saved = True

                return Response(
                    content=content,
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
