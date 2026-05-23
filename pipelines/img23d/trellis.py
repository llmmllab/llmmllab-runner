"""TRELLIS-based image-to-3D pipeline.

Wraps Microsoft TRELLIS (`microsoft/TRELLIS-image-large`) or any compatible
fine-tune (e.g. TencentARC `Pixal3D`) inside the
:class:`InProcessPipeline` contract.

TRELLIS produces multi-view 3D representations (Gaussian splat + mesh) from
a single conditioning image.  The pipeline accepts a base64-encoded image
plus a handful of sampler parameters and returns the rendered preview
images as base64, together with paths to the persisted mesh / splat files.

Heavy dependencies (``torch``, ``trellis``, ``imageio``, ``trimesh``,
``xformers``) are imported lazily inside :meth:`_load`.  This lets the
rest of the runner start without those packages installed — useful in
test and CI environments where the CUDA-extension build is unavailable.

When TRELLIS isn't installed the pipeline raises a clear, structured
``RuntimeError`` from :meth:`_load` describing which package is missing.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import time
import uuid
from typing import Any, Dict, Optional

from config import SD_OUTPUT_DIR
from models import ModelTask
from pipelines.base import InProcessPipeline


# Default repo when the request doesn't specify one.  TencentARC `Pixal3D`
# is built on the same backbone and the loader can be swapped via env var.
_DEFAULT_MODEL_REPO = os.environ.get(
    "TRELLIS_MODEL_REPO", "JeffreyXiang/TRELLIS-image-large"
)
# Where to persist generated .glb / .ply files for HTTP retrieval.
_OUTPUT_DIR = os.environ.get("TRELLIS_OUTPUT_DIR", os.path.join(SD_OUTPUT_DIR, "3d"))


class TrellisPipeline(InProcessPipeline):
    """Image-to-3D pipeline backed by Microsoft TRELLIS."""

    name = "img23d"
    task = ModelTask.IMAGETO3D

    def __init__(self, model_repo: Optional[str] = None) -> None:
        super().__init__()
        self._model_repo = model_repo or _DEFAULT_MODEL_REPO
        # Concrete TRELLIS pipeline instance — set in :meth:`_load`.
        self._impl: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _load(self) -> None:
        """Lazy-load TRELLIS + torch.

        Performs the heavyweight import inside the method so the rest of
        the runner remains usable when TRELLIS isn't installed.
        """
        try:
            # TRELLIS exports a single top-level ``TrellisImageTo3DPipeline``
            # class.  The package isn't on PyPI; it must be installed from
            # the github repo as documented in the README.
            from trellis.pipelines import (  # type: ignore[import-not-found]
                TrellisImageTo3DPipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                "TRELLIS is not installed.  Install with:\n"
                "  pip install git+https://github.com/microsoft/TRELLIS.git\n"
                "and ensure the CUDA toolchain matches torch's CUDA build."
            ) from e

        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        self._logger.info(
            f"Loading TRELLIS weights from {self._model_repo} "
            "(first run will download to ~/.cache/huggingface)"
        )
        self._impl = TrellisImageTo3DPipeline.from_pretrained(self._model_repo)
        try:
            self._impl.cuda()
        except Exception as e:  # noqa: BLE001
            # Fall back to CPU — extremely slow but at least functional in
            # dev environments without a GPU.
            self._logger.warning(
                f"Could not move TRELLIS to CUDA ({e}); using CPU"
            )

    async def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run TRELLIS on one image.

        Payload schema::

            {
                "image_b64": "<base64 PNG/JPEG>",   # required
                "seed":               int,           # optional, default 42
                "ss_steps":           int,           # sparse-structure steps
                "ss_cfg_strength":    float,         # sparse-structure cfg
                "slat_steps":         int,           # SLAT steps
                "slat_cfg_strength":  float,         # SLAT cfg
                "formats":            ["mesh","gaussian"],  # outputs to render
            }

        Response::

            {
                "id":             "<uuid>",
                "preview_b64":    "<base64 PNG of the multi-view preview>",
                "mesh_path":      "/path/to/foo.glb"    # if requested
                "gaussian_path":  "/path/to/foo.ply"    # if requested
                "elapsed_sec":    <float>,
            }
        """
        # ---- Validate payload BEFORE pulling in heavy deps, so callers
        # ---- get a clean 400 even on machines that lack imageio/torch.
        image_b64 = payload.get("image_b64")
        if not image_b64:
            raise ValueError("image_b64 is required")

        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"image_b64 is not valid base64: {e}") from e

        import imageio.v3 as iio  # type: ignore[import-not-found]
        from PIL import Image  # PIL is already a runner dep

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        formats = payload.get("formats", ["mesh"])
        if not isinstance(formats, list) or not formats:
            formats = ["mesh"]

        sampler_params = {
            "steps": int(payload.get("ss_steps", 12)),
            "cfg_strength": float(payload.get("ss_cfg_strength", 7.5)),
        }
        slat_sampler_params = {
            "steps": int(payload.get("slat_steps", 12)),
            "cfg_strength": float(payload.get("slat_cfg_strength", 3.0)),
        }

        started = time.perf_counter()
        outputs = self._impl.run(  # type: ignore[union-attr]
            image,
            seed=int(payload.get("seed", 42)),
            sparse_structure_sampler_params=sampler_params,
            slat_sampler_params=slat_sampler_params,
            formats=formats,
        )

        gen_id = uuid.uuid4().hex[:12]

        result: Dict[str, Any] = {
            "id": gen_id,
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }

        if "mesh" in formats and "mesh" in outputs:
            mesh = outputs["mesh"][0]
            mesh_path = os.path.join(_OUTPUT_DIR, f"{gen_id}.glb")
            mesh.export(mesh_path)
            result["mesh_path"] = mesh_path

        if "gaussian" in formats and "gaussian" in outputs:
            gauss = outputs["gaussian"][0]
            gauss_path = os.path.join(_OUTPUT_DIR, f"{gen_id}.ply")
            gauss.save_ply(gauss_path)
            result["gaussian_path"] = gauss_path

        # Optional multi-view preview render.  TRELLIS' demo notebook calls
        # ``render_utils.render_video`` for this; if the helper isn't
        # available we just skip the preview.
        try:
            from trellis.utils import render_utils  # type: ignore[import-not-found]

            video_frames = render_utils.render_video(
                outputs[formats[0]][0], num_frames=8, resolution=256
            )["color"]
            buf = io.BytesIO()
            iio.imwrite(buf, video_frames[0], extension=".png")
            result["preview_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as e:  # noqa: BLE001
            self._logger.debug(f"Skipping preview render: {e}")

        return result

    async def unload(self) -> None:
        if self._impl is not None:
            try:
                # TRELLIS doesn't expose a clean teardown; drop the
                # reference and let torch's allocator reclaim memory on
                # the next ``torch.cuda.empty_cache()`` call.
                self._impl = None
                try:
                    import torch  # type: ignore[import-not-found]

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                await super().unload()
