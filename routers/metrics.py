"""Prometheus metrics endpoint for the runner."""

from fastapi import APIRouter
from fastapi.responses import Response
from middleware.prometheus_metrics import get_metrics_registry, CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """Expose Prometheus-format metrics."""
    registry = get_metrics_registry()
    body = generate_latest(registry).decode("utf-8")
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)
