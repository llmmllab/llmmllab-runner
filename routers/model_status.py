"""Model status router — exposes model in-use status for external observability.

These endpoints provide model usage telemetry for dashboards, health checks,
and multi-tier orchestration.  Session prioritization is handled at the API
layer (runner_client::_select_runner), so these endpoints are informational
rather than load-bearing.

Endpoints:
    GET /v1/model-status?model_id=X     — single-model status
    GET /v1/model-status/all            — every known model with status
    GET /v1/model-status/available      — models not currently in use
"""

from __future__ import annotations

import time
from datetime import datetime as _dt
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from cache import ServerCache
from utils.model_loader import ModelLoader

router = APIRouter()


def get_server_cache() -> ServerCache:
    """Return a fresh ServerCache instance."""
    return ServerCache()


def get_model_loader() -> ModelLoader:
    """Return a fresh ModelLoader instance (it caches internally)."""
    return ModelLoader()


# ---------------------------------------------------------------------------
# Internal helpers — avoid iterating the server dict multiple times
# ---------------------------------------------------------------------------

def _build_status(
    cache: ServerCache, model_id: str
) -> Dict[str, Any]:
    """Compute in-use status for *model_id* by scanning all servers once.

    Returns a dict with keys: model_id, in_use, active_servers, total_active_servers.
    """
    # Build per-model buckets in one pass over all cached entries.
    # We read through the public stats() API which already holds the lock
    # and returns a snapshot, then filter client-side.
    try:
        all_entries = cache.stats()["servers"]
    except Exception as exc:
        raise HTTPException(500, f"Unable to read server cache: {exc}") from exc

    active_servers: List[Dict[str, Any]] = []
    for entry in all_entries:
        if (entry["model_id"] != model_id
                or entry.get("starting") is True):
            continue
        active_servers.append({
            "server_id": entry["server_id"],
            "port": entry["port"],
            "use_count": entry["use_count"],
            "created_at": _dt.fromtimestamp(entry["created_at"]).isoformat(),
        })

    # Sort by use_count descending (most used first).
    active_servers.sort(key=lambda s: s["use_count"], reverse=True)

    return {
        "model_id": model_id,
        "in_use": len(active_servers) > 0,
        "active_servers": active_servers,
        "total_active_servers": len(active_servers),
    }


def _build_all_statuses(
    cache: ServerCache, loader: ModelLoader
) -> List[Dict[str, Any]]:
    """Build status for every known model in a single pass over the server list.

    Returns a list of dicts with keys: model_id, name, provider, task, in_use,
    active_servers, total_active_servers.
    """
    all_models = loader.get_available_models()
    if not all_models:
        return []

    # One-pass bucketing: group servers by model_id.
    try:
        raw_servers = cache.stats()["servers"]
    except Exception as exc:
        raise HTTPException(500, f"Unable to read server cache: {exc}") from exc

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for entry in raw_servers:
        mid = entry["model_id"]
        if entry.get("starting") is True:
            continue
        buckets.setdefault(mid, []).append({
            "server_id": entry["server_id"],
            "port": entry["port"],
            "use_count": entry["use_count"],
            "created_at": _dt.fromtimestamp(entry["created_at"]).isoformat(),
        })

    model_statuses: List[Dict[str, Any]] = []
    for model_id, model in all_models.items():
        servers = buckets.get(model_id, [])
        servers.sort(key=lambda s: s["use_count"], reverse=True)
        model_statuses.append({
            "model_id": model_id,
            "name": model.name,
            "provider": model.provider,
            "task": model.task.value,
            "in_use": len(servers) > 0,
            "active_servers": servers,
            "total_active_servers": len(servers),
        })

    return model_statuses


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@router.get("/v1/model-status")
def get_model_status(
    model_id: str = Query(..., description="Model ID to check status for"),
):
    """Get the in-use status for a specific model.

    Returns whether the model is currently being used by any active server.
    Primarily useful for external dashboards and health-check integrations.
    (Session routing is handled at the API layer, not via this endpoint.)

    Response:
    - `in_use`: boolean indicating if model is actively being used
    - `active_servers`: list of server entries using this model
    - `total_active_servers`: count of active servers for this model
    """
    cache = get_server_cache()
    loader = get_model_loader()

    # Validate that the model_id is known (optional — still return data even
    # if the model isn't in config; it just means nothing is loaded).
    return _build_status(cache, model_id)


@router.get("/v1/model-status/all")
def get_all_model_status():
    """Get in-use status for all models.

    Returns a list of all known models with their current in-use status.
    Useful for external dashboards and observability tooling.
    """
    cache = get_server_cache()
    loader = get_model_loader()

    model_statuses = _build_all_statuses(cache, loader)
    return {
        "total_models": len(model_statuses),
        "models": model_statuses,
    }


@router.get("/v1/model-status/available")
def get_available_models():
    """Get list of known models that are NOT currently in use.

    Returns models where `in_use` is False.  Useful for external dashboards
    and multi-tier orchestration outside the API service.
    """
    cache = get_server_cache()
    loader = get_model_loader()

    all_statuses = _build_all_statuses(cache, loader)
    available = [s for s in all_statuses if not s["in_use"]]

    return {
        "total_available": len(available),
        "available": [
            {
                **s,
                "active_servers": [],
                "total_active_servers": 0,
            }
            for s in available
        ],
    }
