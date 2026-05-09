"""Server lifecycle router - create, status, delete, release servers."""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config import RUNNER_PORT, SERVER_START_OOM_RETRIES
from task_queue import Priority, RequestSource, parse_priority_header, parse_source_header
from server_manager import LlamaCppServerManager
from utils.hardware_manager import hardware_manager
from utils.logging import llmmllogger
from utils.model_loader import ModelLoader
from middleware.runner_metrics import record_server_start

logger = llmmllogger.bind(component="servers_router")
router = APIRouter()
model_loader = ModelLoader()


class CreateServerRequest(BaseModel):
    model_id: str
    priority: int = 10


def _int_to_priority(value: int) -> Priority:
    """Map legacy integer priority (1-10) to Priority enum.

    Lower integer = higher priority (1 is highest, 10 is lowest).
    1-3 → HIGH, 4-7 → MEDIUM, 8-10 → LOW
    """
    if value <= 3:
        return Priority.HIGH
    elif value <= 7:
        return Priority.MEDIUM
    else:
        return Priority.LOW


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
async def create_server(request: Request, body: CreateServerRequest):
    """Acquire or create a llama.cpp server for the given model.

    Accepts priority information via:
    - Body field `priority` (1-10, lower = higher priority)
    - Header `X-Request-Priority` (high|medium|low) — overrides body
    - Header `X-Request-Source` (user|scheduled|system)

    Reuses an existing healthy server for the model if available.
    If a server is already starting for this model, polls until ready.
    Creates a new one only if none exists and none is starting.
    Returns server_id and base_url for proxying requests.
    """
    from app import server_cache, request_queue

    # Determine priority from headers (override body) or fall back to body value
    header_priority = parse_priority_header(
        request.headers.get("x-request-priority")
    )
    source = parse_source_header(request.headers.get("x-request-source"))

    # Use header priority if explicitly set, otherwise map body integer
    if request.headers.get("x-request-priority"):
        priority = header_priority
    else:
        priority = _int_to_priority(body.priority)

    logger.info(
        f"Server create request for model {body.model_id}: "
        f"priority={priority.name}, source={source.value}"
    )

    # If there's a queue and items ahead of us, wait our turn
    queue_size = await request_queue.size()
    if queue_size > 0:
        logger.info(
            f"Queue has {queue_size} items ahead, enqueuing request "
            f"for model {body.model_id} at priority {priority.name}"
        )
        future = await request_queue.enqueue(
            request=body,
            priority=priority,
            source=source,
        )
        try:
            await asyncio.wait_for(future, timeout=120)
        except asyncio.TimeoutError:
            await request_queue.cancel(future)
            raise HTTPException(
                status_code=503,
                detail="Request timed out waiting in priority queue",
            )
        except asyncio.CancelledError:
            raise HTTPException(
                status_code=503,
                detail="Request was cancelled in priority queue",
            )

    model = model_loader.get_model_by_id(body.model_id)
    if not model:
        raise HTTPException(
            status_code=404, detail=f"Model {body.model_id} not found"
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
        for i in range(240):  # up to 120 seconds (240 * 0.5s)
            await asyncio.sleep(0.5)
            entry = server_cache.acquire_by_model(model.id, log_miss=False)
            if entry is not None:
                base_url = f"http://localhost:{RUNNER_PORT}/v1/server/{entry.server_id}"
                return {
                    "server_id": entry.server_id,
                    "base_url": base_url,
                    "model": model.id,
                    "port": entry.port,
                }
            # Log progress every 10 seconds
            if i > 0 and i % 20 == 0:
                logger.info(
                    f"Still waiting for model {model.id} to be ready ({i * 0.5:.0f}s elapsed)"
                )
        raise HTTPException(
            status_code=500,
            detail=f"Server for model {model.id} failed to become ready",
        )

    # Evict idle servers if needed for VRAM
    _evict_for_vram(model)

    manager = LlamaCppServerManager(
        model=model,
    )

    # Register as "starting" BEFORE starting the process to prevent duplicates
    server_id = server_cache.register_starting(
        model_id=model.id,
        port=manager.port,
        manager=manager,
    )

    # Run blocking start() in a thread pool to avoid blocking the event loop.
    # Retry on transient failures (OOM, segfault, etc.) with exponential backoff:
    # the kernel may need time to reclaim memory or release GPU resources.
    max_retries = SERVER_START_OOM_RETRIES
    last_error = None
    for attempt in range(max_retries + 1):
        started = await asyncio.get_event_loop().run_in_executor(None, manager.start)
        if started:
            break

        # Check for retryable exit codes:
        #   -9  = SIGKILL (OOM killer)
        #   -11 = SIGSEGV (segfault, often GPU driver / VRAM pressure)
        #   -4  = SIGILL  (illegal instruction, sometimes CUDA driver mismatch)
        exit_code = manager.process.returncode if manager.process else None
        is_retryable = exit_code in (-9, -11, -4)

        if not is_retryable or attempt >= max_retries:
            last_error = f"Server start failed (exit code: {exit_code})"
            break

        backoff = 2 ** (attempt + 1)  # 4s, 8s
        reason = "OOM" if exit_code == -9 else "segfault" if exit_code == -11 else "signal"
        logger.warning(
            f"{reason} detected on server start for model {model.id}, "
            f"retrying in {backoff}s (attempt {attempt + 1}/{max_retries})"
        )
        await asyncio.sleep(backoff)

    if not started:
        server_cache.remove(server_id)
        detail = last_error or "Failed to start llama.cpp server"
        raise HTTPException(status_code=500, detail=detail)

    # Mark as ready
    server_cache.mark_ready(server_id)
    record_server_start(model.id)

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
    logger.info(
        f"Force-evicted server {server_id} (model {entry.model_id}, port {entry.port})"
    )

    return {"status": "evicted", "server_id": server_id}
