"""Mesh-to-parts pipeline backed by Tencent Hunyuan3D-Part (XPart + P3-SAM).

Wraps `tencent/Hunyuan3D-Part <https://huggingface.co/tencent/Hunyuan3D-Part>`_
— a separate model from Hunyuan3D-2.1 that takes a *whole* mesh and
decomposes it into semantically meaningful parts (e.g. chair → seat +
legs + back as distinct geometry).

Two stages run end-to-end inside the pipeline:

  1. **P3-SAM** (``p3sam.safetensors``) — predicts part bounding boxes
     from the input mesh.
  2. **XPart** (``model.safetensors`` + ``conditioner.safetensors`` +
     ``shapevae.safetensors``) — regenerates each detected part as a
     standalone, high-fidelity mesh and emits both an assembled
     "decomposed" output and an "exploded" visualization.

Returns paths to four ``.glb`` files on the runner pod:

  * ``mesh_path``       — assembled decomposed mesh (parts joined)
  * ``exploded_path``   — exploded view with parts spatially separated
  * ``bbox_path``       — the bounding-box wireframe only
  * ``gt_bbox_path``    — input mesh overlaid with predicted bboxes
                          (debug view)

The pipeline is **mesh-in / meshes-out** — NOT image-in.  The api wires
this so callers either upload a base64-encoded ``.glb`` directly or pass
a ``mesh_url`` from a prior Hunyuan3D-2.1 ``/v1/images/3d`` run.

Heavy dependencies (``torch``, ``partgen``, ``trimesh``, ``spconv``,
``torch_cluster``) are imported lazily inside :meth:`_load` so the rest
of the runner remains usable on machines that lack the
XPart-specific deps.  Missing-dep failures surface as a clean
``RuntimeError`` from ``_load`` describing how to install them.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
import uuid
from typing import Any, Dict, Optional

from config import SD_OUTPUT_DIR
from models import ModelTask
from pipelines.base import InProcessPipeline


# Yaml drives the on-disk path (``details.model_path`` in
# .models.yaml).  See ``pipelines/rembg/rmbg.py`` for the
# pattern; this is identical except the registry id is
# ``hunyuan3d-part``.
_MODEL_ID = "hunyuan3d-part"

# Where to persist generated .glb outputs for HTTP retrieval.  Same
# root the other 3D pipelines write into; the runner's
# ``/v1/pipelines/img23d_part/files/{filename}`` proxy serves from
# here.
_OUTPUT_DIR = os.environ.get(
    "HUNYUAN3D_PART_OUTPUT_DIR",
    os.path.join(SD_OUTPUT_DIR, "3d_parts"),
)

# Default octree resolution for marching cubes.  Higher = finer mesh
# detail but quadratically more memory + time.  512 matches the demo
# script; 256 is faster for iteration if needed.
_DEFAULT_OCTREE_RESOLUTION = int(
    os.environ.get("HUNYUAN3D_PART_OCTREE_RESOLUTION", "512")
)

_INSTALL_HINT = (
    "Hunyuan3D-Part dependencies are missing.  The pipeline needs the\n"
    "vendored XPart package, plus its CUDA-bound deps:\n"
    "  pip install spconv-cu124 fpsample addict easydict scikit-learn\n"
    "  pip install -e vendors/Hunyuan3D-Part/XPart\n"
    "and Sonata weights from facebook/sonata (downloaded automatically\n"
    "on first load if HF_TOKEN is set)."
)


class Hunyuan3DPartPipeline(InProcessPipeline):
    """Mesh-to-parts pipeline backed by Tencent Hunyuan3D-Part (XPart)."""

    name = "img23d_part"
    task = ModelTask.IMAGETO3D

    #: Identifier in ``.models.yaml`` — used to look up
    #: ``details.model_path`` lazily on first ``_load``.
    model_id: str = _MODEL_ID

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        # ``model_path`` is the on-disk directory containing
        # ``model/``, ``conditioner/``, ``shapevae/``, ``p3sam/``,
        # ``scheduler/``.  When None, resolved from ``.models.yaml``
        # at first load.  Tests pass an explicit path.
        self._model_path: Optional[str] = model_path
        self._impl: Any = None
        # Set by ``_load`` to ``cuda:N`` (N = freest GPU) or ``cpu``.
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """Look up ``details.model_path`` from the model registry."""
        from utils.model_loader import ModelLoader  # local — avoid import cycle

        loader = ModelLoader()
        model = loader.get_model_by_id(self.model_id)
        if model is None:
            raise RuntimeError(
                f"Hunyuan3D-Part pipeline could not find '{self.model_id}' "
                f"in the model registry (.models.yaml).  Add an entry with "
                f"``details.model_path`` pointing at the on-disk weights "
                f"directory."
            )
        path = getattr(model.details, "model_path", None)
        if not path:
            raise RuntimeError(
                f"Hunyuan3D-Part pipeline: model '{self.model_id}' has no "
                f"``details.model_path`` set in .models.yaml.  This field "
                f"is required for in_process pipelines."
            )
        return path

    async def _load(self) -> None:
        """Lazy-load XPart's PartFormerPipeline + torch."""
        try:
            import torch  # type: ignore[import-not-found]
            from partgen.partformer_pipeline import (  # type: ignore[import-not-found]
                PartFormerPipeline,
            )
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e

        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        if self._model_path is None:
            self._model_path = self._resolve_model_path()

        # XPart needs >12 GB VRAM (the multi-model architecture loads
        # conditioner + UNet + VAE + P3-SAM + Sonata + bbox predictor
        # simultaneously).  The runner box has a mix of 3060 (12 GB)
        # and 3090s (24 GB) — defaulting to ``cuda:0`` lands on the
        # 3060 and OOMs at load time.  Pick the device with the most
        # free VRAM at load time instead.
        device = "cpu"
        if torch.cuda.is_available():
            best_idx = 0
            best_free = -1
            for i in range(torch.cuda.device_count()):
                free, _total = torch.cuda.mem_get_info(i)
                if free > best_free:
                    best_free = free
                    best_idx = i
            device = f"cuda:{best_idx}"
            # Also pin the process's default cuda device to ``best_idx``
            # so any submodule that constructs tensors with bare
            # ``cuda`` (no explicit index) lands on the same GPU.  Without
            # this, XPart's conditioner ends up on cuda:0 while
            # P3-SAM lands on best_idx and the first cross-module
            # tensor op trips NCCL (multi-GPU transfer) with an
            # "unhandled system error".
            torch.cuda.set_device(best_idx)
            self._logger.info(
                f"Selected {device} for XPart "
                f"({best_free / 1024**3:.1f} GB free); "
                f"torch.cuda default device set to {best_idx}"
            )

        self._logger.info(
            f"Loading Hunyuan3D-Part (XPart) from {self._model_path}"
        )
        # XPart's ``smart_load_model`` joins HY3DGEN_MODELS + model_path.
        # We pass an absolute path to make the resolution deterministic
        # regardless of HY3DGEN_MODELS env value.
        self._impl = PartFormerPipeline.from_pretrained(
            model_path=self._model_path,
            dtype=torch.float32,
            device=device,
        )

        # XPart requires fp32 for stability (spconv kernels + the
        # bbox predictor are float32-only paths).  No bf16/fp16
        # downcast here unlike rembg/Hunyuan3D-2.1's shape pass.
        try:
            if device.startswith("cuda"):
                self._impl.to(device=device, dtype=torch.float32)
        except Exception as e:  # noqa: BLE001
            self._logger.warning(
                f"Could not move Hunyuan3D-Part to {device} ({e}); using CPU"
            )
        # Remember the chosen device so _run can move input tensors to it.
        self._device = device

    async def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run XPart on one input mesh.

        Payload schema::

            {
                "mesh_b64":      "<base64-encoded .glb>",   # required
                "octree_resolution": int,                     # optional, default 512
                "seed":          int,                         # optional
            }

        Response::

            {
                "id":             "<uuid>",
                "elapsed_sec":    <float>,
                "mesh_path":      "/data/sd-out/3d_parts/<id>_decomposed.glb",
                "exploded_path":  "/data/sd-out/3d_parts/<id>_exploded.glb",
                "bbox_path":      "/data/sd-out/3d_parts/<id>_bbox.glb",
                "gt_bbox_path":   "/data/sd-out/3d_parts/<id>_gt_bbox.glb",
            }

        The api wraps these with ``mesh_url`` / ``exploded_url`` /
        ``bbox_url`` / ``gt_bbox_url`` fields that route through
        ``GET /v1/images/3d/parts/{filename}``.
        """
        mesh_b64 = payload.get("mesh_b64")
        if not mesh_b64:
            raise ValueError("mesh_b64 is required")

        try:
            mesh_bytes = base64.b64decode(mesh_b64, validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"mesh_b64 is not valid base64: {e}") from e

        # XPart wants a file path, not bytes — its loader uses trimesh.load
        # under the hood which goes through the file extension.  Use a
        # temp file in the output dir so the mesh stays accessible if we
        # need to debug a failed run.
        gen_id = uuid.uuid4().hex[:12]
        input_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_input.glb")
        with open(input_path, "wb") as f:
            f.write(mesh_bytes)

        octree_resolution = _DEFAULT_OCTREE_RESOLUTION
        try:
            if "octree_resolution" in payload:
                octree_resolution = int(payload["octree_resolution"])
        except (TypeError, ValueError):
            pass

        kwargs: Dict[str, Any] = {}
        if "seed" in payload:
            try:
                kwargs["seed"] = int(payload["seed"])
            except (TypeError, ValueError):
                pass

        started = time.perf_counter()
        try:
            # XPart returns (obj_mesh, (out_bbox, mesh_gt_bbox, explode_object)).
            obj_mesh, bbox_tuple = self._impl(  # type: ignore[misc]
                mesh_path=input_path,
                octree_resolution=octree_resolution,
                output_type="trimesh",
                **kwargs,
            )
            out_bbox, mesh_gt_bbox, explode_object = bbox_tuple
        finally:
            # Best-effort clean up of the temp input.  Keep it on
            # exception so the user can inspect what was sent.
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except OSError:
                pass

        # Persist all four outputs side-by-side so the api can serve
        # whichever the caller asked for.
        mesh_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_decomposed.glb")
        exploded_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_exploded.glb")
        bbox_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_bbox.glb")
        gt_bbox_path = os.path.join(_OUTPUT_DIR, f"{gen_id}_gt_bbox.glb")

        obj_mesh.export(mesh_path)
        explode_object.export(exploded_path)
        out_bbox.export(bbox_path)
        mesh_gt_bbox.export(gt_bbox_path)

        return {
            "id": gen_id,
            "elapsed_sec": round(time.perf_counter() - started, 2),
            "mesh_path": mesh_path,
            "exploded_path": exploded_path,
            "bbox_path": bbox_path,
            "gt_bbox_path": gt_bbox_path,
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
