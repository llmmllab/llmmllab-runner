"""Model status router - exposes model in-use status for multi-runner session prioritization."""

from typing import Any, Dict, List
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException

from cache import ServerCache
from utils.model_loader import ModelLoader

router = APIRouter()
server_cache = None
_model_loader = None


def get_server_cache():
    """Lazy singleton for ServerCache."""
    global server_cache
    if server_cache is None:
        from cache import ServerCache
        server_cache = ServerCache()
    return server_cache


def get_model_loader():
    """Lazy singleton for ModelLoader."""
    global _model_loader
    if _model_loader is None:
        from utils.model_loader import ModelLoader
        _model_loader = ModelLoader()
    return _model_loader


@router.get("/v1/model-status")
def get_model_status(
    model_id: str = Query(..., description="Model ID to check status for"),
):
    """Get the in-use status for a specific model.

    Returns whether the model is currently being used by any active server.
    Useful for API clients to determine which runner to route sessions to
    when the same model exists on multiple runners.

    Response:
    - `in_use`: boolean indicating if model is actively being used
    - `active_servers`: list of server entries using this model
    - `total_active_servers`: count of active servers for this model
    """
    cache = get_server_cache()
    loader = get_model_loader()

    # Get all servers for this model
    active_servers = []
    for entry in cache._servers.values():
        if entry.model_id == model_id and not entry.starting:
            # Check if server is healthy and running
            if entry.manager is not None and hasattr(entry.manager, "is_running"):
                if entry.manager.is_running():
                    active_servers.append({
                        "server_id": entry.server_id,
                        "port": entry.port,
                        "use_count": entry.use_count,
                        "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
                    })

    # Sort by use_count descending (most used first)
    active_servers.sort(key=lambda x: x["use_count"], reverse=True)

    return {
        "model_id": model_id,
        "in_use": len(active_servers) > 0,
        "active_servers": active_servers,
        "total_active_servers": len(active_servers),
    }


@router.get("/v1/model-status/all")
def get_all_model_status():
    """Get in-use status for all models.

    Returns a list of all models with their current in-use status.
    Useful for API clients to discover which models are available and which are busy.
    """
    cache = get_server_cache()
    loader = get_model_loader()

    # Get all available models
    all_models = loader.get_available_models()

    # Build status for each model
    model_statuses = []
    for model_id, model in all_models.items():
        status = get_model_status(model_id=model_id)
        model_statuses.append({
            "model_id": model_id,
            "name": model.name,
            "provider": model.provider,
            "task": model.task.value,
            **status,
        })

    return {
        "total_models": len(model_statuses),
        "models": model_statuses,
    }


@router.get("/v1/model-status/available")
def get_available_models():
    """Get list of models that are NOT currently in use.

    Returns models where in_use is False. Useful for API clients to
    discover which runners have available capacity for new sessions.
    """
    cache = get_server_cache()
    loader = get_model_loader()

    # Get all available models
    all_models = loader.get_available_models()

    # Filter to only models that are not in use
    available = []
    for model_id, model in all_models.items():
        status = get_model_status(model_id=model_id)
        if not status["in_use"]:
            available.append({
                "model_id": model_id,
                "name": model.name,
                "provider": model.provider,
                "task": model.task.value,
                "active_servers": [],
                "total_active_servers": 0,
            })

    return {
        "total_available": len(available),
        "available": available,
    }
