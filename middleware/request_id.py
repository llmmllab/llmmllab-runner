"""Generate and propagate X-Request-ID for traceability across services."""

import uuid
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Inject a request ID into every request and echo it in the response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Inject trace_id into structlog context for log correlation
        ctx = trace.get_current_span().get_span_context()
        if ctx and ctx.is_valid:
            request.state.trace_id = f"{ctx.trace_id:032x}"

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
