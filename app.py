"""llmmllab-runner - Standalone llama.cpp server manager and proxy."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI

from config import RUNNER_HOST, RUNNER_PORT, DCGM_METRICS_INTERVAL_SEC, LLAMA_METRICS_INTERVAL_SEC
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
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="RunnerApp")

# Global cache instance (initialized at startup)
server_cache: ServerCache = None  # type: ignore
_evict_task: Optional[asyncio.Task] = None
_metrics_task: Optional[asyncio.Task] = None

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
    global server_cache, _evict_task, _metrics_task
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

    yield

    logger.info("Runner shutting down")
    # Cancel the background tasks before stopping servers
    for task in (_evict_task, _metrics_task):
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
