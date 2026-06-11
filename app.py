"""llmmllab-runner - Standalone llama.cpp server manager and proxy."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI

from config import (
    RUNNER_HOST,
    RUNNER_PORT,
    DCGM_METRICS_INTERVAL_SEC,
    LLAMA_METRICS_INTERVAL_SEC,
    SLOT_SAVE_DIR,
    SLOT_CLEANUP_MAX_AGE_MIN,
    SLOT_CLEANUP_MAX_SIZE_MB,
    SLOT_CLEANUP_INTERVAL_SEC,
    SLOT_INACTIVE_MAX_AGE_MIN,
)
from cache import ServerCache
from routers import models as models_router
from routers import servers as servers_router
from routers import metrics as metrics_router
from routers import status as status_router
from routers import pipelines as pipelines_router
from routers import model_status as model_status_router
from proxy import router as proxy_router
from middleware import RequestIdMiddleware, PrometheusMiddleware
from middleware.runner_metrics import (
    update_server_metrics,
    update_gpu_metrics,
    update_dcgm_metrics,
    update_llama_server_metrics,
)
from middleware.tracing import setup_tracing, shutdown_tracing
from utils.hardware_manager import hardware_manager
from utils.logging import llmmllogger, set_session_id_ctx, reset_session_id_ctx

logger = llmmllogger.bind(component="RunnerApp")

# Global cache instance (initialized at startup)
server_cache: ServerCache = None  # type: ignore
_evict_task: Optional[asyncio.Task] = None
_metrics_task: Optional[asyncio.Task] = None
_slot_cleanup_task: Optional[asyncio.Task] = None

# Module-level ModelLoader singleton (avoids re-instantiation on every health check)
_model_loader = None


def get_model_loader():
    """Lazy singleton for ModelLoader."""
    global _model_loader
    if _model_loader is None:
        from utils.model_loader import ModelLoader
        _model_loader = ModelLoader()
    return _model_loader


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global server_cache, _evict_task, _metrics_task, _slot_cleanup_task
    logger.info("Runner starting up")
    server_cache = ServerCache()
    logger.info("ServerCache initialized")

    # Start periodic eviction task
    async def evict_idle_servers():
        while True:
            await asyncio.sleep(60)
            evicted = server_cache.evict_idle()
            if evicted:
                logger.info(f"Evicted {len(evicted)} idle servers")
            hardware_manager.check_gpu_thermals()
            # Update Prometheus metrics
            try:
                update_server_metrics(server_cache)
                update_gpu_metrics()
            except Exception as e:
                logger.debug(f"Metrics update failed: {e}")

    _evict_task = asyncio.create_task(evict_idle_servers())

    # Start periodic DCGM + llama.cpp metrics scraping task
    async def scrape_extended_metrics():
        while True:
            await asyncio.sleep(DCGM_METRICS_INTERVAL_SEC)
            try:
                # DCGM GPU metrics
                await update_dcgm_metrics()
                # Llama.cpp server metrics
                if server_cache:
                    await update_llama_server_metrics(server_cache)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Extended metrics scrape failed: {e}")

    _metrics_task = asyncio.create_task(scrape_extended_metrics())

    # Start periodic slot file cleanup task
    async def cleanup_slot_files():
        import glob as glob_mod
        import time as time_mod

        while True:
            await asyncio.sleep(SLOT_CLEANUP_INTERVAL_SEC)
            if not SLOT_SAVE_DIR:
                continue

            try:
                slot_files = glob_mod.glob(f"{SLOT_SAVE_DIR}/*.bin")
                now = time_mod.time()
                max_age_sec = SLOT_CLEANUP_MAX_AGE_MIN * 60
                inactive_sec = SLOT_INACTIVE_MAX_AGE_MIN * 60
                deleted = 0
                total_freed = 0

                # Import session activity tracker from proxy router
                from proxy.router import get_session_activity

                # Delete slot files for inactive sessions.
                # For tracked sessions, use the proxy's activity timestamp.
                # For untracked (orphaned) sessions, fall back to file mtime.
                if inactive_sec > 0:
                    for filepath in sorted(slot_files):
                        try:
                            # Extract session_id from filename:
                            # slot_{session_id}.bin (legacy) or
                            # slot_{session_id}_{server_id}.bin (model-specific)
                            basename = os.path.basename(filepath)
                            if not basename.startswith("slot_") or not basename.endswith(
                                ".bin"
                            ):
                                continue
                            name_part = basename[5:-4]  # strip "slot_" and ".bin"
                            # If server_id is embedded, strip it to get session_id
                            # Server IDs are UUIDs appended after the last underscore
                            # but session_ids can also contain underscores, so we use
                            # get_session_activity to check validity.
                            session_id = name_part
                            last_active = get_session_activity(session_id)

                            # Use session activity if available, else file mtime
                            if last_active is not None:
                                idle_since = last_active
                            else:
                                idle_since = os.path.getmtime(filepath)

                            if now - idle_since > inactive_sec:
                                size = os.path.getsize(filepath)
                                os.remove(filepath)
                                deleted += 1
                                total_freed += size
                        except OSError:
                            pass

                # Absolute floor: delete any file older than SLOT_CLEANUP_MAX_AGE_MIN
                # regardless of session activity (catches orphaned files)
                if max_age_sec > 0:
                    for filepath in sorted(slot_files):
                        try:
                            mtime = os.path.getmtime(filepath)
                            if now - mtime > max_age_sec:
                                size = os.path.getsize(filepath)
                                os.remove(filepath)
                                deleted += 1
                                total_freed += size
                        except OSError:
                            pass

                # Enforce total size limit
                if SLOT_CLEANUP_MAX_SIZE_MB > 0:
                    target_bytes = SLOT_CLEANUP_MAX_SIZE_MB * 1024 * 1024
                    while True:
                        try:
                            remaining = glob_mod.glob(f"{SLOT_SAVE_DIR}/*.bin")
                            total_size = sum(
                                os.path.getsize(f) for f in remaining
                                if os.path.isfile(f)
                            )
                            if total_size <= target_bytes:
                                break
                            oldest = min(
                                remaining, key=lambda f: os.path.getmtime(f)
                            )
                            size = os.path.getsize(oldest)
                            os.remove(oldest)
                            deleted += 1
                            total_freed += size
                        except (OSError, ValueError):
                            break

                if deleted:
                    logger.info(
                        "Slot cleanup complete",
                        deleted=deleted,
                        freed_mb=round(total_freed / 1024 / 1024, 2),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Slot cleanup failed: {e}")

    _slot_cleanup_task = asyncio.create_task(cleanup_slot_files())

    yield

    logger.info("Runner shutting down")
    # Cancel the background tasks before stopping servers
    for task in (_evict_task, _metrics_task, _slot_cleanup_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    # Save every active slot's KV to disk BEFORE we close upstream
    # connections.  This is the SIGTERM half of the aggressive-persistence
    # design — every session that has a pinned slot at shutdown time
    # leaves a fresh file in SLOT_SAVE_DIR, so the next pod start can
    # restore it on the session's first request.  Bounded by the
    # container's terminationGracePeriodSeconds (default 30s in our
    # deployment.yaml); each save is one HTTP call to llama.cpp's
    # /slots/N?action=save endpoint, typically a few hundred ms.
    try:
        from proxy.router import save_all_active_slots
        save_count = await save_all_active_slots()
        logger.info(
            "Persisted active slots on shutdown",
            extra={"saved": save_count},
        )
    except Exception as e:
        logger.warning(f"save_all_active_slots during shutdown failed: {e}")

    # Close the pooled httpx clients used by the proxy.  Background tasks
    # above may have held connections; close after they're cancelled but
    # before we kill the upstream llama.cpp processes so any in-flight
    # requests get a clean transport-level signal.
    try:
        from proxy.router import aclose_all_clients
        await aclose_all_clients()
    except Exception as e:
        logger.warning(f"aclose_all_clients during shutdown failed: {e}")
    if server_cache:
        server_cache.stop_all()
    # Shutdown tracing
    shutdown_tracing()
    logger.info("Runner shutdown complete")


app = FastAPI(
    title="llmmllab-runner",
    description="Standalone llama.cpp server manager and request proxy",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount routers
app.include_router(models_router.router)
app.include_router(servers_router.router)
app.include_router(proxy_router.router)
app.include_router(metrics_router.router)
app.include_router(status_router.router)
app.include_router(pipelines_router.router)
app.include_router(model_status_router.router)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(PrometheusMiddleware)

class SessionIdMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            from starlette.datastructures import Headers
            headers = Headers(scope=scope)
            session_id = headers.get("X-Session-ID")
            token = set_session_id_ctx(session_id)
            try:
                await self.app(scope, receive, send)
            finally:
                reset_session_id_ctx(token)
        else:
            await self.app(scope, receive, send)

app.add_middleware(SessionIdMiddleware)

# Initialize distributed tracing
setup_tracing("llmmllab-runner", app)


@app.get("/health")
def health():
    """Health check endpoint.

    Returns per-GPU stats keyed by gpu id under ``gpu`` (legacy shape,
    e.g. ``{"0": {"free_mb": ...}, "1": ...}``) AND an aggregate
    ``available_vram_bytes`` summed across all GPUs.  The api's runner
    selection prefers the per-GPU shape so it can compute *effective*
    free VRAM by intersecting with a model's ``tensor_split`` — a model
    pinned to device 0 via ``tensor_split: "1,0,0"`` shouldn't be
    credited with VRAM that lives on devices 1 and 2.
    """
    gpu_stats = hardware_manager.gpu_stats()
    aggregate_free_bytes = 0
    for entry in gpu_stats.values():
        if isinstance(entry, dict) and "free_mb" in entry:
            try:
                aggregate_free_bytes += int(float(entry["free_mb"]) * 1024 * 1024)
            except (TypeError, ValueError):
                pass

    models = []
    try:
        loader = get_model_loader()
        for m in loader.get_available_models().values():
            models.append({
                "id": m.id,
                "name": m.name,
                "task": str(m.task),
            })
    except Exception:
        pass

    active = 0
    if server_cache:
        active = len(server_cache.stats().get("servers", []))

    return {
        "status": "ok",
        "gpu": {
            **gpu_stats,
            "available_vram_bytes": aggregate_free_bytes,
        },
        "active_servers": active,
        "models": models,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=RUNNER_HOST,
        port=RUNNER_PORT,
        reload=False,
    )
