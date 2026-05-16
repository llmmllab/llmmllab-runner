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
)
from cache import ServerCache
from routers import models as models_router
from routers import servers as servers_router
from routers import metrics as metrics_router
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
                deleted = 0
                total_freed = 0

                # Delete files older than SLOT_CLEANUP_MAX_AGE_MIN
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
    """Health check endpoint."""
    gpu_stats = hardware_manager.gpu_stats()
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
        "gpu": gpu_stats,
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
