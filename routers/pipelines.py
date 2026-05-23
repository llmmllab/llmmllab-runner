"""Router for in-process pipelines (TRELLIS, future HF/diffusers stacks).

Subprocess-backed runtimes (llama.cpp, stable-diffusion.cpp) get a server
process per model and route through ``proxy/router.py``.  Pipelines that
cannot ship a standalone server live entirely inside this process and are
exposed via ``/v1/pipelines/<name>/<endpoint>``.

The set of pipelines is registered once at startup in :data:`_REGISTRY`.
A pipeline lazy-loads its weights on first request — failure surfaces as
HTTP 503 with a structured error explaining what's missing.

API contract::

    POST /v1/pipelines/<name>/run
    {
        ...payload (pipeline-specific)...
    }

    -> 200 { ...result (pipeline-specific)... }

    GET  /v1/pipelines
    -> 200 { "pipelines": [{"name": "img23d", "task": "ImageTo3D", "loaded": false}, ...] }

    POST /v1/pipelines/<name>/unload
    -> 200 { "name": "img23d", "loaded": false }
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from pipelines.base import InProcessPipeline
from pipelines.img23d.trellis import TrellisPipeline
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="pipelines_router")
router = APIRouter(prefix="/v1/pipelines", tags=["pipelines"])

# Single-instance registry, populated at import time.  Keeping this static
# (rather than a runtime ``Dict[str, type[InProcessPipeline]]`` lookup) means
# the FastAPI app surface is fully described by code review — no
# late-bound registration paths to chase.
_REGISTRY: Dict[str, InProcessPipeline] = {
    TrellisPipeline.name: TrellisPipeline(),
}


def _get_pipeline(name: str) -> InProcessPipeline:
    pipeline = _REGISTRY.get(name)
    if pipeline is None:
        raise HTTPException(
            status_code=404,
            detail={
                "reason": "pipeline_not_found",
                "message": f"Pipeline '{name}' is not registered on this runner",
                "available_pipelines": sorted(_REGISTRY.keys()),
            },
        )
    return pipeline


@router.get("")
def list_pipelines() -> Dict[str, List[Dict[str, Any]]]:
    """List every pipeline known to the runner and its loaded state."""
    return {
        "pipelines": [
            {
                "name": p.name,
                "task": p.task.value if p.task is not None else None,
                "loaded": p.loaded,
            }
            for p in _REGISTRY.values()
        ]
    }


@router.post("/{name}/run")
async def run_pipeline(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a pipeline.  Lazy-loads weights on the first call."""
    pipeline = _get_pipeline(name)
    try:
        return await pipeline.run(payload)
    except ValueError as e:
        # Validation error from the pipeline's payload check.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        # Loading failed — usually a missing optional dependency.
        logger.error(f"Pipeline {name} failed to load: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "reason": "pipeline_unavailable",
                "message": str(e),
            },
        ) from e


@router.post("/{name}/unload")
async def unload_pipeline(name: str) -> Dict[str, Any]:
    """Release GPU memory held by a loaded pipeline."""
    pipeline = _get_pipeline(name)
    await pipeline.unload()
    return {"name": name, "loaded": pipeline.loaded}
