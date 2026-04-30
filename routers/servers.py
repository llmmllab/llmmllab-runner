"""Server lifecycle router - create, status, delete, release servers."""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import RUNNER_PORT
from models import UserConfig
from server_manager import LlamaCppServerManager
from utils.hardware_manager import hardware_manager
from utils.logging import llmmllogger
from utils.model_loader import ModelLoader

logger = llmmllogger.bind(component="servers_router")
router = APIRouter()
model_loader = ModelLoader()


class CreateServerRequest(BaseModel):
    model_id: str
    priority: int = 10
    config_override: Optional[Dict[str, Any]] = None


def _estimate_model_size(model) -> float:
    """Estimate VRAM needed for a model, in bytes."""
    try:
        size = model.details.size
        if size > 0:
            return size + (128 * 1024 * 1024)  # 128 MB overhead
    except Exception:
        pass
    return 4 * 1024 * 1024 * 1024  # 4 GB fallback


def _evict_for_vram(model):
    """Evict idle servers until enough VRAM is available for the model."""
    required_bytes = _estimate_model_size(model)
    available = hardware_manager.available_vram_bytes()
    if available >= required_bytes:
        return

    from app import server_cache

    eligible = server_cache.get_eligible_for_eviction()
    for entry in eligible:
        if entry.manager is not None:
            try:
                entry.manager.stop()
            except Exception:
                pass
        server_cache.remove(entry.server_id)
        available = hardware_manager.available_vram_bytes()
        if available >= required_bytes:
            break


@router.post("/v1/server/create")
async def create_server(request: CreateServerRequest):
    """Acquire or create a llama.cpp server for the given model.

    Reuses an existing healthy server for the model if available.
    If a server is already starting for this model, polls until ready.
    Creates a new one only if none exists and none is starting.
    Returns server_id and base_url for proxying requests.
    """
    from app import server_cache

    model = model_loader.get_model_by_id(request.model_id)
    if not model:
        raise HTTPException(
            status_code=404, detail=f"Model {request.model_id} not found"
        )

    assert model.id
    # Try to acquire an existing healthy server first
    entry = server_cache.acquire_by_model(model.id)
    if entry is not None:
        base_url = f"http://localhost:{RUNNER_PORT}/v1/server/{entry.server_id}"
        return {
            "server_id": entry.server_id,
            "base_url": base_url,
            "model": model.id,
            "port": entry.port,
        }

    # If a server is already starting for this model, wait for it
    if server_cache.has_starting_server(model.id):
        logger.info(
            f"Server already starting for model {model.id}, waiting for readiness..."
        )
        for _ in range(240):  # up to 120 seconds (120 * 2 polls per second)
            await asyncio.sleep(0.5)
            entry = server_cache.acquire_by_model(model.id)
            if entry is not None:
                base_url = f"http://localhost:{RUNNER_PORT}/v1/server/{entry.server_id}"
                return {
                    "server_id": entry.server_id,
                    "base_url": base_url,
                    "model": model.id,
                    "port": entry.port,
                }
        raise HTTPException(
            status_code=500,
            detail=f"Server for model {model.id} failed to become ready",
        )

    # Evict idle servers if needed for VRAM
    _evict_for_vram(model)

    user_config = None
    if request.config_override:
        try:
            user_config = UserConfig(**request.config_override)
        except Exception:
            user_config = None

    manager = LlamaCppServerManager(
        model=model,
        user_config=user_config,
    )

    # Register as "starting" BEFORE starting the process to prevent duplicates
    server_id = server_cache.register_starting(
        model_id=model.id,
        port=manager.port,
        manager=manager,
    )

    # Run blocking start() in a thread pool to avoid blocking the event loop
    started = await asyncio.get_event_loop().run_in_executor(None, manager.start)
    if not started:
        server_cache.remove(server_id)
        raise HTTPException(status_code=500, detail="Failed to start llama.cpp server")

    # Mark as ready
    server_cache.mark_ready(server_id)

    base_url = f"http://localhost:{RUNNER_PORT}/v1/server/{server_id}"

    return {
        "server_id": server_id,
        "base_url": base_url,
        "model": model.id,
        "port": manager.port,
    }


@router.get("/v1/server/{server_id}")
def get_server(server_id: str):
    """Get status of a running server."""
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    is_running = entry.manager.is_running() if entry.manager else False

    return {
        "server_id": entry.server_id,
        "model_id": entry.model_id,
        "port": entry.port,
        "use_count": entry.use_count,
        "healthy": entry.healthy,
        "running": is_running,
        "created_at": entry.created_at,
    }


@router.delete("/v1/server/{server_id}")
def delete_server(server_id: str):
    """Stop and remove a server."""
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    if entry.manager is not None:
        entry.manager.stop()

    server_cache.remove(server_id)

    return {"status": "deleted", "server_id": server_id}


@router.post("/v1/server/{server_id}/release")
def release_server(server_id: str):
    """Decrement use count for a server (signal that a client is done)."""
    from app import server_cache

    ok = server_cache.decrement_use(server_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    entry = server_cache.get(server_id)
    return {
        "server_id": server_id,
        "use_count": entry.use_count if entry else 0,
    }


@router.post("/v1/server/{server_id}/evict")
def evict_server(server_id: str):
    """Force-evict a server regardless of idle state.

    Immediately stops the llama-server process and removes it from the cache.
    This is an optional manual override — normal eviction is automatic based
    on idle time.
    """
    from app import server_cache

    entry = server_cache.get(server_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    if entry.manager is not None:
        try:
            entry.manager.stop()
        except Exception as e:
            logger.warning(f"Error stopping server {server_id}: {e}")

    server_cache.remove(server_id)
    logger.info(f"Force-evicted server {server_id} (model {entry.model_id}, port {entry.port})")

    return {"status": "evicted", "server_id": server_id}
