"""llmmllab-runner - Standalone llama.cpp server manager and proxy."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI

from config import RUNNER_HOST, RUNNER_PORT
from cache import ServerCache
from routers import models as models_router
from routers import servers as servers_router
from proxy import router as proxy_router
from utils.hardware_manager import hardware_manager
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="RunnerApp")

# Global cache instance (initialized at startup)
server_cache: ServerCache = None  # type: ignore
_evict_task: Optional[asyncio.Task] = None

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
    global server_cache, _evict_task
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

    _evict_task = asyncio.create_task(evict_idle_servers())

    yield

    logger.info("Runner shutting down")
    # Cancel the background eviction task before stopping servers
    if _evict_task:
        _evict_task.cancel()
        try:
            await _evict_task
        except asyncio.CancelledError:
            pass
    if server_cache:
        server_cache.stop_all()
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
