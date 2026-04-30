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

router = APIRouter()


async def _stream_upstream(client, method, url, headers, body, server_id):
    """Stream response from upstream, keeping client open for the entire duration.

    Returns a StreamingResponse that maintains the upstream connection
    until the downstream client is done consuming. Decrements the server's
    use count when the stream fully completes or is abandoned.
    """

    response = await client.send(
        client.build_request(
            method=method,
            url=url,
            headers=headers,
            content=body if body else None,
        ),
        stream=True,
    )
    response_headers = dict(response.headers)
    status_code = response.status_code
    clean_headers = {k: v for k, v in response_headers.items()
                     if k.lower() not in ("transfer-encoding", "content-length")}

    async def upstream_iterator():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            response.close()
            await client.aclose()
            # Request fully consumed (or client disconnected) — mark server idle
            from app import server_cache
            server_cache.decrement_use(server_id)

    return StreamingResponse(
        content=upstream_iterator(),
        status_code=status_code,
        headers=clean_headers,
    )


@router.api_route("/v1/server/{server_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
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

    # Mark server as in-use during request
    server_cache.increment_use(server_id)
    is_streaming = False

    try:
        target_host = f"http://127.0.0.1:{entry.port}"

        # Rewrite path: strip the /v1/server/{id} prefix
        remaining = path
        upstream_url = f"{target_host}/{remaining}"

        # Read request body
        body = await request.body()

        # Build headers (exclude hop-by-hop)
        hop_by_hop = {
            "host", "connection", "keep-alive", "transfer-encoding",
            "upgrade", "proxy-authenticate", "proxy-authorization",
            "te", "trailers", "proxy-connection",
        }
        headers = dict(request.headers)
        for h in hop_by_hop:
            headers.pop(h, None)
        headers["host"] = f"127.0.0.1:{entry.port}"

        method = request.method
        client = httpx.AsyncClient(timeout=120.0)

        # Check if this is likely an SSE request (POST to /v1/chat/completions)
        is_likely_sse = (
            method == "POST"
            and "/chat/completions" in remaining
            and body
        )

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
                client, method, upstream_url, headers, body, server_id,
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
                    headers={k: v for k, v in dict(response.headers).items()
                             if k.lower() not in ("transfer-encoding", "content-length")},
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
        if not is_streaming:
            server_cache.decrement_use(server_id)
