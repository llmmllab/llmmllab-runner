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

    GET  /v1/pipelines/img23d/files/<filename>
    -> 200 <binary> (model/gltf-binary or application/octet-stream)
"""

import os
import re
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import SD_OUTPUT_DIR
from pipelines.base import InProcessPipeline
from pipelines.img23d.hunyuan3d import Hunyuan3DPipeline
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="pipelines_router")
router = APIRouter(prefix="/v1/pipelines", tags=["pipelines"])

# Single-instance registry, populated at import time.  Keeping this static
# (rather than a runtime ``Dict[str, type[InProcessPipeline]]`` lookup) means
# the FastAPI app surface is fully described by code review — no
# late-bound registration paths to chase.
_REGISTRY: Dict[str, InProcessPipeline] = {
    Hunyuan3DPipeline.name: Hunyuan3DPipeline(),
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


# ---------------------------------------------------------------------------
# Static file serving for TRELLIS outputs
# ---------------------------------------------------------------------------
#
# TRELLIS writes .glb mesh files and .ply gaussian-splat files into
# ``SD_OUTPUT_DIR/3d/``.  ``POST /v1/pipelines/img23d/run`` returns those
# absolute paths in its response (mesh_path / gaussian_path), and the api
# layer's ``GET /v1/images/3d/{filename}`` proxies through to this
# endpoint so clients can download the artefacts without needing pod
# access.
#
# We hard-restrict the served filenames to ``<hex>.{glb,ply,png}`` —
# generation IDs are uuid hex prefixes, so anything else is suspicious
# and would only be exploitable as a path-traversal vector.

_IMG23D_OUTPUT_DIR = os.environ.get(
    "TRELLIS_OUTPUT_DIR", os.path.join(SD_OUTPUT_DIR, "3d")
)
_IMG23D_FILENAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\.(glb|ply|png)$")

_IMG23D_CONTENT_TYPES = {
    ".glb": "model/gltf-binary",
    ".ply": "application/octet-stream",
    ".png": "image/png",
}


@router.get("/img23d/files/{filename}")
def download_img23d_artifact(filename: str) -> FileResponse:
    """Serve a TRELLIS output file by basename.

    Filename must match ``<id>.{glb,ply,png}`` where ``<id>`` is a short
    alphanumeric token.  Anything containing path separators, dots, or
    other characters is rejected with 400 to keep the endpoint
    traversal-safe.

    Returns the file with a content-type appropriate to its extension
    (``model/gltf-binary`` for .glb so glTF viewers recognise it).
    """
    if not _IMG23D_FILENAME_RE.match(filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid filename '{filename}'. "
                "Expected <id>.{glb,ply,png} with alphanumeric id."
            ),
        )

    file_path = os.path.join(_IMG23D_OUTPUT_DIR, filename)
    # Defence-in-depth: even though the regex already blocks traversal,
    # confirm the resolved path stays inside the output directory.
    real_dir = os.path.realpath(_IMG23D_OUTPUT_DIR)
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(real_dir + os.sep) and real_path != real_dir:
        raise HTTPException(status_code=400, detail="Path traversal blocked")

    if not os.path.isfile(real_path):
        raise HTTPException(
            status_code=404,
            detail=f"Artefact '{filename}' not found on this runner",
        )

    ext = os.path.splitext(filename)[1].lower()
    media_type = _IMG23D_CONTENT_TYPES.get(ext, "application/octet-stream")

    return FileResponse(
        real_path,
        media_type=media_type,
        filename=filename,
    )
