"""HTTP proxy router - forwards requests to llama.cpp servers.

Catch-all route for /v1/server/{server_id}/* that rewrites the path
and forwards to the appropriate local llama.cpp server instance.

Tracks server activity: increments use_count on request start, decrements
when the response fully completes. This allows the cache eviction timer to
fire based on actual request activity rather than external release calls.
"""

import json

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from config import PROXY_TIMEOUT
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="proxy_router")

router = APIRouter()


async def _stream_upstream(client, method, url, headers, body, server_id):
    """Stream response from upstream, keeping client open for the entire duration.

    Returns a StreamingResponse that maintains the upstream connection
    until the downstream client is done consuming. Decrements the server's
    use count when the stream fully completes or is abandoned.

    When the downstream client disconnects mid-stream, ``aclose()`` is
    called on the upstream response which closes the TCP connection to
    llama.cpp, causing it to stop generating tokens for the abandoned slot.
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
        # Upstream crashed, refused connection, or timed out before sending
        # any response headers (e.g., OOM, segfault, killed by the kernel).
        # Decrement use_count here because the caller's finally block skips
        # it when is_streaming is True.
        logger.error(
            "Upstream server %s error before response: %s", server_id, exc
        )
        await client.aclose()
        from app import server_cache
        server_cache.decrement_use(server_id)
        raise
    except Exception as exc:
        # Catch-all for connection errors, timeouts, or other transport failures
        # before the response headers arrive.
        logger.error(
            "Upstream server %s failed before response: %s", server_id, exc
        )
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
        stream_error = None
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        except httpx.RemoteProtocolError as exc:
            # Incomplete chunked read: upstream closed mid-stream.
            # This can happen due to OOM, context overflow, or network issues.
            stream_error = exc
            logger.error(
                "Upstream server %s closed connection mid-stream: %s", server_id, exc
            )
        except Exception as exc:
            # Other stream errors.
            stream_error = exc
            logger.error(
                "Upstream server %s stream error: %s", server_id, exc
            )
        finally:
            # Close the upstream response (aborts connection if still open,
            # which signals llama.cpp to stop generating for this slot)
            try:
                await response.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass
            # Request fully consumed (or client disconnected) — mark server idle
            from app import server_cache
            server_cache.decrement_use(server_id)
        # Re-raise the error so the caller knows the stream failed
        if stream_error:
            raise stream_error from None

    return StreamingResponse(
        content=upstream_iterator(),
        status_code=status_code,
        headers=clean_headers,
    )


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
    """
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    target_host = f"http://127.0.0.1:{entry.port}"

    # Reject chat completion requests when all slots are busy to prevent
    # request queueing that causes cascading timeouts with --parallel 1.
    # Include a Retry-After header so callers can back off instead of
    # hammering the endpoint in a tight retry loop.
    # Only reject if the server is healthy (responding to /health) AND
    # all slots report as busy — a freshly restarted server may show stale
    # slot state, so we skip the check if /health fails.
    remaining = path
    if "chat/completions" in remaining:
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

    # Mark server as in-use during request
    server_cache.increment_use(server_id)
    is_streaming = False

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
        is_likely_sse = method == "POST" and "/chat/completions" in remaining and body

        if is_likely_sse:
            try:
                body_dict = json.loads(body)
                is_likely_sse = body_dict.get("stream", False)
            except Exception:
                is_likely_sse = True  # be safe, stream if we can't parse

        if is_likely_sse:
            # Streaming: decrement happens in upstream_iterator's finally block
            is_streaming = True
            return await _stream_upstream(
                client,
                method,
                upstream_url,
                headers,
                body,
                server_id,
            )

        # Non-streaming: buffer entire response
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
    except Exception as exc:
        # Catch-all for any other unexpected errors during proxying.
        logger.error(
            "Upstream server %s error: %s", server_id, exc
        )
        raise HTTPException(
            status_code=503,
            detail=f"Upstream server {server_id} failed: {exc}",
        )
    finally:
        # Decrement for non-streaming requests and errors.
        # Streaming requests decrement inside _stream_upstream's iterator
        # finally block when the stream fully drains or client disconnects.
        if not is_streaming:
            server_cache.decrement_use(server_id)
