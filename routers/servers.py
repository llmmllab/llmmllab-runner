"""Server lifecycle router - create, status, delete, release servers."""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import time

from config import (
    RUNNER_PORT,
    SERVER_START_OOM_RETRIES,
    VRAM_EVICT_MIN_RESIDENCY_SEC,
    VRAM_EVICT_RELEASE_WAIT_SEC,
)
from models import ModelProvider
from server_manager import LlamaCppServerManager, SDCppServerManager
from utils.hardware_manager import hardware_manager
from utils.logging import llmmllogger, _session_id_ctx
from utils.model_loader import ModelLoader
from middleware.runner_metrics import record_server_start

# Error reason codes for structured error responses
MODEL_NOT_CONFIGURED = "model_not_configured"
MODEL_NOT_AVAILABLE = "model_not_available"
INSUFFICIENT_RESOURCES = "insufficient_resources"
SERVER_START_FAILED = "server_start_failed"

logger = llmmllogger.bind(component="servers_router")
router = APIRouter()
model_loader = ModelLoader()


def _build_error_response(
    reason: str,
    message: str,
    *,
    requested_model: Optional[str] = None,
    available_models: Optional[List[str]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured error response with context for debugging.

    Returns a dict suitable for use as the ``detail`` field in HTTPException.
    The structure is:
        {
            "reason": <error_code>,
            "message": <human_readable>,
            "requested_model": <model_id>,
            "available_models": [<list of model IDs on this runner>],
            "details": {<extra context>}
        }
    """
    response = {
        "reason": reason,
        "message": message,
    }
    if requested_model is not None:
        response["requested_model"] = requested_model
    if available_models is not None:
        response["available_models"] = available_models
    if details is not None:
        response["details"] = details
    return response


def _get_available_model_ids() -> List[str]:
    """Return a sorted list of model IDs configured on this runner."""
    return sorted(model_loader.get_available_models().keys())


class CreateServerRequest(BaseModel):
    model_id: str
    num_ctx: Optional[int] = None


def _estimate_model_size(model) -> float:
    """Estimate VRAM needed for a model, in bytes.

    Sums the weights (``details.size``) plus the multimodal projector
    (``details.clip_model_path``, the mmproj GGUF) when present, because the
    clip/mmproj is loaded into VRAM alongside the weights but is NOT included
    in ``details.size``.  For the small-runner vision models this is a
    non-trivial 1.3-1.8 GB of F32 mmproj that the old VRAM-only estimate
    ignored — under-counting the footprint so ``_evict_for_vram`` freed too
    little and let two models try to co-load on the single 12 GB card,
    triggering the cudaMalloc co-load OOM.  We only ever ADD to the estimate
    here, so the change is conservative (more eviction headroom, never less).

    KV cache is intentionally NOT added: these models set ``kv_on_cpu: True``
    so the KV lands in host RAM, not VRAM — counting it against VRAM would be
    wrong.  (Host-RAM accounting is governed by the pod's cgroup memory limit
    instead; see k8s/deployment.yaml.)
    """
    import os

    overhead = 128 * 1024 * 1024  # 128 MB
    try:
        size = model.details.size
        if not size or size <= 0:
            return 4 * 1024 * 1024 * 1024  # 4 GB fallback

        # Add the mmproj/clip projector VRAM footprint when present.
        clip_path = getattr(model.details, "clip_model_path", None)
        if clip_path:
            try:
                size += os.path.getsize(clip_path)
            except OSError:
                # File missing / unreadable — fall back to a conservative
                # 1.5 GB allowance (typical F32 mmproj) rather than ignore it.
                size += int(1.5 * 1024 * 1024 * 1024)

        return size + overhead
    except Exception:
        return 4 * 1024 * 1024 * 1024  # 4 GB fallback


def _wait_for_vram_release(
    required_bytes: float,
    *,
    baseline: float,
    timeout_sec: float,
) -> float:
    """Poll free VRAM until it reaches ``required_bytes`` or grows past
    ``baseline``, up to ``timeout_sec``.  Returns the last observed free VRAM.

    ``manager.stop()`` sends SIGTERM asynchronously — nvidia-smi keeps
    reporting the dying process's VRAM as resident until the kernel reaps it.
    Re-reading free VRAM immediately therefore under-counts what we just
    freed, which made the old eviction loop either evict more servers than
    necessary or (worse) conclude it had freed nothing and 500 anyway.  We
    give the driver a moment, returning as soon as either (a) we have enough
    for the new model, or (b) VRAM has visibly grown beyond the pre-stop
    baseline (the freed process has been reaped, so further waiting on THIS
    eviction is pointless — the caller should evict the next candidate).
    """
    deadline = time.monotonic() + max(0.0, timeout_sec)
    available = hardware_manager.available_vram_bytes()
    while time.monotonic() < deadline:
        available = hardware_manager.available_vram_bytes()
        if available >= required_bytes or available > baseline:
            return available
        time.sleep(0.25)
    return hardware_manager.available_vram_bytes()


def _evict_for_vram(model) -> bool:
    """Evict least-recently-used IDLE servers until ``model`` fits in VRAM.

    Returns ``True`` if, after any eviction, free VRAM is at least the model's
    estimated footprint (i.e. the create can proceed), ``False`` if even after
    evicting every eligible idle server there still isn't room (the caller
    surfaces a retryable insufficient-resources error rather than letting the
    llama-server start OOM into a 500).

    Eviction policy (see ``ServerCache.get_idle_lru_for_vram`` /
    ``VRAM_EVICT_MIN_RESIDENCY_SEC``):
      * only fully-idle servers (``use_count == 0``) — never one with
        in-flight requests;
      * only those idle past a short min-residency — avoids load/evict
        thrashing;
      * coldest (longest-idle) first.

    After each ``stop()`` we wait for the driver to actually reclaim the VRAM
    before deciding whether to evict the next candidate, so we free the
    minimum necessary.
    """
    required_bytes = _estimate_model_size(model)
    available = hardware_manager.available_vram_bytes()
    if available >= required_bytes:
        return True

    from app import server_cache

    candidates = server_cache.get_idle_lru_for_vram(VRAM_EVICT_MIN_RESIDENCY_SEC)
    if not candidates:
        logger.warning(
            f"VRAM pressure for model {model.id}: need "
            f"{required_bytes / 1024**3:.1f} GB, have "
            f"{available / 1024**3:.1f} GB free, but NO idle server is "
            f"eligible for eviction (all busy or within min-residency)."
        )
        return False

    for entry in candidates:
        baseline = hardware_manager.available_vram_bytes()
        logger.info(
            f"VRAM pressure for model {model.id}: evicting idle LRU server "
            f"{entry.server_id} (model {entry.model_id}, idle_since="
            f"{entry.idle_since}) to free space "
            f"(need {required_bytes / 1024**3:.1f} GB, "
            f"have {baseline / 1024**3:.1f} GB)."
        )
        if entry.manager is not None:
            try:
                # Mark intentional so the watchdog doesn't log this as a crash
                # or emit a process_died metric.
                if hasattr(entry.manager, "_intentional_stop"):
                    entry.manager._intentional_stop = True
                entry.manager.stop()
            except Exception as e:
                logger.warning(
                    f"Error stopping server {entry.server_id} during VRAM "
                    f"eviction: {e}"
                )
        server_cache.remove(entry.server_id)
        try:
            from middleware.runner_metrics import record_server_eviction

            record_server_eviction("vram_pressure")
        except Exception:
            pass
        available = _wait_for_vram_release(
            required_bytes,
            baseline=baseline,
            timeout_sec=VRAM_EVICT_RELEASE_WAIT_SEC,
        )
        if available >= required_bytes:
            logger.info(
                f"VRAM eviction freed enough for model {model.id}: "
                f"{available / 1024**3:.1f} GB free."
            )
            return True

    logger.warning(
        f"VRAM eviction for model {model.id} evicted all "
        f"{len(candidates)} eligible idle server(s) but only "
        f"{available / 1024**3:.1f} GB free of "
        f"{required_bytes / 1024**3:.1f} GB needed."
    )
    return False


@router.post("/v1/server/create")
async def create_server(request: Request, body: CreateServerRequest):
    """Acquire or create a llama.cpp server for the given model.

    Reuses an existing healthy server for the model if available.
    If a server is already starting for this model, polls until ready.
    Creates a new one only if none exists and none is starting.
    Returns server_id and base_url for proxying requests.
    """
    from app import server_cache

    logger.info(f"Server create request for model {body.model_id}")

    model = model_loader.get_model_by_id(body.model_id)
    if not model:
        available = _get_available_model_ids()
        detail = _build_error_response(
            reason=MODEL_NOT_CONFIGURED,
            message=f"Model '{body.model_id}' is not configured on this runner",
            requested_model=body.model_id,
            available_models=available if len(available) <= 20 else None,
        )
        logger.warning(
            f"Model not configured: '{body.model_id}'. "
            f"Available models on this runner: {len(available)}"
        )
        raise HTTPException(status_code=404, detail=detail)

    assert model.id

    # Guard: refuse to start a server when the requested context exceeds
    # what the model can provide. The runner knows n_ctx from the model
    # definition, so it can reject undersized requests before spinning up
    # a llama.cpp process.
    requested_ctx = body.num_ctx
    model_ctx = (model.parameters.num_ctx if model.parameters else None) or 90000
    if requested_ctx is not None and requested_ctx > model_ctx:
        logger.warning(
            f"Context too large: requested={requested_ctx}, "
            f"model_ctx={model_ctx} for model {model.id}"
        )
        raise HTTPException(
            status_code=507,
            detail={
                "reason": "context_too_large",
                "message": (
                    f"Requested context size ({requested_ctx} tokens) exceeds "
                    f"the model's configured context window ({model_ctx} tokens). "
                    f"Reduce num_ctx or use a model with a larger context window."
                ),
                "requested_model": model.id,
                "details": {
                    "requested_num_ctx": requested_ctx,
                    "model_num_ctx": model_ctx,
                },
            },
        )

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
            detail=_build_error_response(
                reason=SERVER_START_FAILED,
                message=f"Server for model '{model.id}' failed to become ready within 120s",
                requested_model=model.id,
            ),
        )

    # Evict idle servers if needed to make room for this model.  When even
    # evicting every eligible idle server can't free enough VRAM, fail fast
    # with a structured 507 (insufficient capacity) rather than spawning a
    # llama-server that would OOM into a 500 — the api treats 507 as
    # "this runner is full, try a peer / requeue" WITHOUT tripping the
    # circuit breaker, so the request survives instead of taking the runner
    # offline for everyone.
    if not _evict_for_vram(model):
        available_vram = hardware_manager.available_vram_bytes()
        model_size = _estimate_model_size(model)
        logger.warning(
            f"Insufficient VRAM for model {model.id} after eviction: "
            f"need {model_size / 1024**3:.1f} GB, have "
            f"{available_vram / 1024**3:.1f} GB free."
        )
        raise HTTPException(
            status_code=507,
            detail=_build_error_response(
                reason=INSUFFICIENT_RESOURCES,
                message=(
                    f"Insufficient VRAM to start a server for model "
                    f"'{model.id}': all GPUs are occupied by in-use servers "
                    f"that cannot be evicted. Estimated model size: "
                    f"{model_size / (1024**3):.1f} GB, available VRAM: "
                    f"{available_vram / (1024**3):.1f} GB."
                ),
                requested_model=model.id,
                details={
                    "estimated_model_size_bytes": int(model_size),
                    "available_vram_bytes": int(available_vram),
                },
            ),
        )

    session_id = _session_id_ctx.get() or request.headers.get("x-session-id")

    # Dispatch to the right native runtime based on the model's declared
    # provider.  llama.cpp handles text/embeddings; stable-diffusion.cpp
    # handles image generation (txt2img / img2img).  IN_PROCESS pipelines
    # (Hunyuan3D, RMBG, …) don't have a subprocess to acquire — they live
    # inside this Python process and are exposed at /v1/pipelines/*.
    # New subprocess-backed providers slot in here without touching the
    # proxy or cache.
    if model.provider == ModelProvider.STABLE_DIFFUSION_CPP:
        manager = SDCppServerManager(model=model, session_id=session_id)
    elif model.provider == ModelProvider.IN_PROCESS:
        raise HTTPException(
            status_code=400,
            detail=_build_error_response(
                reason="in_process_pipeline",
                message=(
                    f"Model '{model.id}' is an in-process pipeline — "
                    f"there is no subprocess server to acquire.  Call "
                    f"POST /v1/pipelines/<pipeline_name>/run directly "
                    f"(see GET /v1/pipelines)."
                ),
                requested_model=model.id,
            ),
        )
    else:
        manager = LlamaCppServerManager(model=model, session_id=session_id)

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
        reason = (
            "OOM" if exit_code == -9 else "segfault" if exit_code == -11 else "signal"
        )
        logger.warning(
            f"{reason} detected on server start for model {model.id}, "
            f"retrying in {backoff}s (attempt {attempt + 1}/{max_retries})"
        )
        await asyncio.sleep(backoff)

    if not started:
        server_cache.remove(server_id)

        # Provide a structured error with resource context
        exit_code = None
        if manager.process:
            exit_code = manager.process.returncode

        if exit_code in (-9, -11):
            # OOM or segfault — likely insufficient resources
            available_vram = hardware_manager.available_vram_bytes()
            model_size = _estimate_model_size(model)
            detail = _build_error_response(
                reason=INSUFFICIENT_RESOURCES,
                message=(
                    f"Failed to start server for model '{model.id}': "
                    f"insufficient resources (exit code {exit_code}). "
                    f"Estimated model size: {model_size / (1024**3):.1f} GB, "
                    f"available VRAM: {available_vram / (1024**3):.1f} GB"
                ),
                requested_model=model.id,
                details={
                    "exit_code": exit_code,
                    "estimated_model_size_bytes": int(model_size),
                    "available_vram_bytes": int(available_vram),
                    "retries_attempted": max_retries,
                },
            )
        else:
            detail = _build_error_response(
                reason=SERVER_START_FAILED,
                message=last_error
                or f"Failed to start llama.cpp server for model '{model.id}'",
                requested_model=model.id,
                details={"exit_code": exit_code} if exit_code is not None else {},
            )

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


@router.get("/v1/servers")
def list_servers():
    """List every server currently in the cache (any state).

    Mirrors ``server_cache.stats()`` — returns ``active_servers`` plus a
    flat list of every entry's id, model, port, use_count, etc.  Useful
    for operations tooling (e.g. the force-shutdown script) that needs
    to enumerate servers without knowing each id up-front; ``/health``
    only reports the count.
    """
    from app import server_cache

    if server_cache is None:
        return {"active_servers": 0, "servers": []}
    return server_cache.stats()


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
