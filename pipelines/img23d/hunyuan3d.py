"""Hunyuan3D-2.1-based image-to-3D pipeline.

Wraps Tencent's `Hunyuan3D-2.1 <https://huggingface.co/tencent/Hunyuan3D-2.1>`_
inside the :class:`InProcessPipeline` contract.  The shape-only path needs
~6 GB VRAM, which fits comfortably on a 3060 — leaving the 24 GB 3090s
free for the SD models.

The pipeline takes a single conditioning image (PNG/JPEG, base64) and
returns:

  * ``mesh_path``  — path to a ``.glb`` mesh on the runner pod
  * ``elapsed_sec``— wall-clock time for the run

We deliberately do not expose Hunyuan3D's texture-paint stage at this
layer — it triples VRAM and round-trip time, and the mesh-only output
is enough for the api's existing ``CreateImageTo3DResponse`` contract.
Add it later if the paint route becomes useful.

Heavy dependencies (``torch``, ``hy3dgen``, ``trimesh``) are imported
lazily inside :meth:`_load` so the rest of the runner can start even
when the Hunyuan3D wheel isn't present (development machines, CI).
Missing-dep failures surface as a clean ``RuntimeError`` from ``_load``.
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


# Where to load weights from.  Two-knob layout matches what hy3dgen's
# ``smart_load_model`` expects:
#
#   final path = $HY3DGEN_MODELS / <model_path> / <subfolder>
#
# On the cluster the runner pod mounts the cluster's pre-downloaded
# weights at ``/models/hunyuan3d/`` (containing ``hunyuan3d-dit-v2-1/``
# directly), so we set ``HY3DGEN_MODELS=/models`` and pass
# ``model_path='hunyuan3d'`` — final path resolves to
# ``/models/hunyuan3d/hunyuan3d-dit-v2-1/`` which is where the ``.ckpt``
# + ``config.yaml`` live.
#
# In dev (no pre-downloaded weights), set HY3DGEN_MODELS to
# ``~/.cache/hy3dgen`` (hy3dgen's default) and the canonical HF repo id
# ``tencent/Hunyuan3D-2.1`` — hy3dgen will snapshot_download into that
# layout on first load.
_DEFAULT_MODEL_PATH = os.environ.get(
    "HUNYUAN3D_MODEL_PATH", "tencent/Hunyuan3D-2.1"
)
_DEFAULT_SUBFOLDER = os.environ.get(
    "HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-1"
)
# When the weights are stored as ``.ckpt`` (the format on this cluster's
# PVC) rather than ``.safetensors`` (HF default), flip this off.
_USE_SAFETENSORS = os.environ.get(
    "HUNYUAN3D_USE_SAFETENSORS", "false"
).lower() in ("1", "true", "yes", "on")
# Where to persist generated .glb meshes for HTTP retrieval.  Same root
# directory the SD pipelines write into; the api's
# ``/v1/pipelines/img23d/files/{filename}`` proxy serves from here.
_OUTPUT_DIR = os.environ.get(
    "TRELLIS_OUTPUT_DIR", os.path.join(SD_OUTPUT_DIR, "3d")
)

_INSTALL_HINT = (
    "Hunyuan3D-2.1 is not installed.  Install with:\n"
    "  git clone https://github.com/Tencent/Hunyuan3D-2 /opt/hunyuan3d\n"
    "  cd /opt/hunyuan3d && pip install -r requirements.txt && pip install -e .\n"
    "  cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && cd ../../..\n"
    "  cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install\n"
    "and ensure the CUDA toolchain matches torch's CUDA build."
)


class Hunyuan3DPipeline(InProcessPipeline):
    """Image-to-3D pipeline backed by Tencent Hunyuan3D-2.1."""

    name = "img23d"
    task = ModelTask.IMAGETO3D

    def __init__(
        self,
        model_path: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._subfolder = subfolder or _DEFAULT_SUBFOLDER
        # Concrete pipeline instance — set in :meth:`_load`.
        self._impl: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _load(self) -> None:
        """Lazy-load Hunyuan3D + torch.

        Heavy-import inside the method so the rest of the runner remains
        usable when Hunyuan3D isn't installed.
        """
        try:
            from hy3dgen.shapegen import (  # type: ignore[import-not-found]
                Hunyuan3DDiTFlowMatchingPipeline,
            )
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e

        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        self._logger.info(
            f"Loading Hunyuan3D-2.1 from "
            f"$HY3DGEN_MODELS/{self._model_path}/{self._subfolder} "
            f"(use_safetensors={_USE_SAFETENSORS})"
        )
        self._impl = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self._model_path,
            subfolder=self._subfolder,
            use_safetensors=_USE_SAFETENSORS,
        )

        # Move to CUDA when available.  Hunyuan3D's pipeline exposes
        # ``.to(device)`` (it's a torch.nn.Module-ish wrapper); fall
        # back to CPU if GPU placement throws — useful in dev envs
        # without a GPU.
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                self._impl.to("cuda")
        except Exception as e:  # noqa: BLE001
            self._logger.warning(
                f"Could not move Hunyuan3D to CUDA ({e}); using CPU"
            )

    async def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run Hunyuan3D on one conditioning image.

        Payload schema (subset of the legacy TRELLIS schema — extra
        fields are accepted but ignored so the api's
        ``CreateImageTo3DRequest`` stays unchanged)::

            {
                "image_b64":   "<base64 PNG/JPEG>",   # required
                "seed":        int,                    # optional, default 42
                "num_inference_steps": int,            # optional, default 30
                "guidance_scale":      float,          # optional, default 5.5
                "formats":     ["mesh"],               # informational only —
                                                       # Hunyuan3D-2.1 mesh-only
            }

        Response::

            {
                "id":           "<uuid>",
                "mesh_path":    "/data/sd-out/3d/<id>.glb",
                "gaussian_path": null,    # Hunyuan3D does not produce splats
                "preview_b64":  null,     # no auto preview
                "elapsed_sec":  <float>,
            }
        """
        # ---- Validate payload BEFORE the heavy imports so a malformed
        #      request returns 400 even on machines that lack torch/PIL.
        image_b64 = payload.get("image_b64")
        if not image_b64:
            raise ValueError("image_b64 is required")

        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"image_b64 is not valid base64: {e}") from e

        from PIL import Image  # PIL is already a runner dep

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        kwargs: Dict[str, Any] = {}
        if "seed" in payload:
            try:
                seed = int(payload["seed"])
                # Hunyuan3D accepts a generator or `seed=` kwarg
                # depending on version; pass `generator=` only if torch
                # available, else fall through with no-op.
                try:
                    import torch  # type: ignore[import-not-found]

                    kwargs["generator"] = torch.Generator(
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    ).manual_seed(seed)
                except Exception:
                    pass
            except (TypeError, ValueError):
                pass
        if "num_inference_steps" in payload:
            try:
                kwargs["num_inference_steps"] = int(payload["num_inference_steps"])
            except (TypeError, ValueError):
                pass
        if "guidance_scale" in payload:
            try:
                kwargs["guidance_scale"] = float(payload["guidance_scale"])
            except (TypeError, ValueError):
                pass

        started = time.perf_counter()
        # Hunyuan3DDiTFlowMatchingPipeline returns a list of trimesh
        # objects.  We take the first one.
        outputs = self._impl(image=image, **kwargs)  # type: ignore[misc]
        if not outputs:
            raise RuntimeError("Hunyuan3D returned an empty result")
        mesh = outputs[0]

        gen_id = uuid.uuid4().hex[:12]
        mesh_path = os.path.join(_OUTPUT_DIR, f"{gen_id}.glb")
        mesh.export(mesh_path)

        return {
            "id": gen_id,
            "elapsed_sec": round(time.perf_counter() - started, 2),
            "mesh_path": mesh_path,
            # Kept for response-shape parity with the legacy TRELLIS
            # pipeline.  Hunyuan3D-2.1 shape-only path doesn't render
            # gaussian splats or a quick preview frame.
            "gaussian_path": None,
            "preview_b64": None,
        }

    async def unload(self) -> None:
        if self._impl is not None:
            try:
                self._impl = None
                try:
                    import torch  # type: ignore[import-not-found]

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                await super().unload()
