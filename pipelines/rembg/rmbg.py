"""briaai/RMBG-2.0 background-removal pipeline.

A lightweight (~880 MiB) BiRefNet-based segmentation model that produces
a high-quality alpha mask for the subject of an input image.  Unlike
Qwen-Image-Edit, this is a purpose-built segmentation network — it does
exactly one job (foreground/background separation) and does it very
well, including on the kinds of scenes Qwen-Image-Edit refuses to
modify (cup-on-a-table, person against a wall, etc.).

The model loads via transformers' ``trust_remote_code=True`` mechanism
because briaai/RMBG-2.0 ships its own BiRefNet implementation in
``birefnet.py`` rather than registering through the standard
AutoModelForImageSegmentation factory.  Once loaded, inference is a
single forward pass over a 1024×1024 transform of the input.

Payload (POST /v1/pipelines/rembg/run)::

    {
      "image_b64": "<base64 PNG or JPEG>",  # required
      "size":      1024,                     # optional, square edge in
                                             # pixels for the model input;
                                             # defaults to 1024 to match
                                             # the BiRefNet-2.0 recipe.
                                             # Mask is upscaled back to
                                             # the source resolution.
      "mask_only": false                     # optional; if true, skip the
                                             # alpha-composite step and
                                             # return just the mask.
    }

Response::

    {
      "id":              "<uuid>",
      "transparent_b64": "<base64 PNG with alpha — subject on transparent>"  | null,
      "mask_b64":        "<base64 PNG grayscale of the alpha channel>",
      "width":           <int>,   # output dimensions (== input)
      "height":          <int>,
      "elapsed_sec":     <float>,
    }

Heavy deps (``transformers``, ``torch``, ``Pillow``, ``torchvision``)
are imported lazily inside :meth:`_load` so the rest of the runner can
start when this pipeline is unavailable (CI / dev machines).
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


# Yaml is the source of truth.  We resolve the on-disk path lazily on
# first ``_load`` via ``ModelLoader.get_model_by_id("rmbg-2.0")`` and
# read ``details.model_path``.  This keeps the runner deployment env
# free of duplicate path config and lets ops point at a different model
# directory by editing .models.yaml only.
_MODEL_ID = "rmbg-2.0"

# ``model_path`` is a path config field, not an output destination.
# Keep ``RMBG_OUTPUT_DIR`` as an env override because the output dir is
# pure runtime concern (where cached cutout PNGs live).
_OUTPUT_DIR = os.environ.get(
    "RMBG_OUTPUT_DIR", os.path.join(SD_OUTPUT_DIR, "rembg")
)
_DEFAULT_INPUT_SIZE = int(os.environ.get("RMBG_INPUT_SIZE", "1024"))


_INSTALL_HINT = (
    "briaai/RMBG-2.0 dependencies are missing.  The pipeline needs:\n"
    "  pip install transformers torch torchvision Pillow\n"
    "transformers loads the model via trust_remote_code=True because\n"
    "RMBG-2.0 ships its own BiRefNet implementation."
)


class RMBGPipeline(InProcessPipeline):
    """Background-removal pipeline backed by briaai/RMBG-2.0."""

    name = "rembg"
    task = ModelTask.IMAGETOIMAGE

    #: Identifier in ``.models.yaml`` — used to look up
    #: ``details.model_path`` lazily on first ``_load``.
    model_id: str = _MODEL_ID

    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        # If an explicit override is passed, honour it (tests do this).
        # Otherwise the path is resolved from .models.yaml in ``_load``.
        self._model_path: Optional[str] = model_path
        self._impl: Any = None
        self._transform: Any = None
        self._device: str = "cpu"

    def _resolve_model_path(self) -> str:
        """Look up ``details.model_path`` from the model registry."""
        from utils.model_loader import ModelLoader  # local — avoid import cycle

        loader = ModelLoader()
        model = loader.get_model_by_id(self.model_id)
        if model is None:
            raise RuntimeError(
                f"RMBG pipeline could not find '{self.model_id}' in the "
                f"model registry (.models.yaml).  Add an entry with "
                f"``details.model_path`` pointing at the on-disk "
                f"safetensors directory."
            )
        path = getattr(model.details, "model_path", None)
        if not path:
            raise RuntimeError(
                f"RMBG pipeline: model '{self.model_id}' has no "
                f"``details.model_path`` set in .models.yaml.  This "
                f"field is required for in_process pipelines."
            )
        return path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _load(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForImageSegmentation,
            )
            from torchvision import transforms  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(_INSTALL_HINT) from e

        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        # Resolve the path from the model registry on first load (unless
        # the constructor was given an explicit override, which is what
        # the tests do).
        if self._model_path is None:
            self._model_path = self._resolve_model_path()

        self._logger.info(
            f"Loading briaai/RMBG-2.0 from {self._model_path}"
        )
        # trust_remote_code is required — the repo bundles its own
        # BiRefNet implementation in birefnet.py.  Safe here because the
        # model_path is operator-controlled config, not user input.
        self._impl = AutoModelForImageSegmentation.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self._impl.eval()

        # Hardware placement is yaml-driven (.models.yaml ``parameters``:
        # ``main_gpu`` + ``tensor_split``).  RMBG is tiny (~1.6 GB
        # fp16) so it fits anywhere; default ``main_gpu: -1`` picks
        # the freest card and avoids piling on top of an LLM's KV
        # cache when there's headroom elsewhere.
        from pipelines._gpu_select import pick_device, device_hints_from_model

        try:
            from utils.model_loader import ModelLoader
            model = ModelLoader().get_model_by_id(self.model_id)
        except Exception:
            model = None
        choice = pick_device(
            **device_hints_from_model(model),
            min_vram_gb=2.0,  # RMBG fp16 ~1.6 GB + workspace
            logger=self._logger,
        )
        try:
            if choice.primary.startswith("cuda"):
                self._device = choice.primary
                self._impl.to(choice.primary)
                # fp16, not bf16 — BiRefNet uses torchvision's deformable
                # convolutions and ``deformable_im2col`` only got a bf16
                # CUDA kernel in torchvision 0.21+.  This image pins to
                # torchvision 0.20.1 (Hunyuan3D's requirement set), so
                # bf16 raises ``"deformable_im2col" not implemented for
                # 'BFloat16'``.  fp16 has had a kernel since 0.15 and is
                # BiRefNet's recommended inference precision — ~2×
                # throughput vs fp32 and half the VRAM.
                self._impl.to(torch.float16)
            else:
                self._device = "cpu"
        except Exception as e:  # noqa: BLE001
            self._logger.warning(
                f"Could not move RMBG to {choice.primary} ({e}); using CPU"
            )
            self._device = "cpu"

        # ImageNet normalisation — the recipe baked into the BiRefNet
        # config.  Resize to a 1024-square then normalize; mask is
        # bilinearly upsampled back to source resolution on the way out.
        self._transform = transforms.Compose([
            transforms.Resize((_DEFAULT_INPUT_SIZE, _DEFAULT_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    async def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_b64 = payload.get("image_b64")
        if not image_b64:
            raise ValueError("image_b64 is required")
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"image_b64 is not valid base64: {e}") from e

        mask_only = bool(payload.get("mask_only", False))

        # Heavy imports already pulled in via _load; safe to use here.
        import torch  # type: ignore[import-not-found]
        from PIL import Image  # PIL is a runner dep

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size

        started = time.perf_counter()
        input_tensor = self._transform(image).unsqueeze(0).to(self._device)  # type: ignore[union-attr]
        # Match model dtype.  ``self._device`` used to be the bare
        # string ``"cuda"`` but the yaml-driven picker now hands us
        # ``"cuda:1"`` etc., so we check the prefix instead of equality.
        # Without this fp16 cast the input stays fp32 while the model
        # is fp16 and pytorch raises
        # ``RuntimeError: Input type (float) and bias type (c10::Half)
        # should be the same``.
        if self._device.startswith("cuda"):
            input_tensor = input_tensor.to(torch.float16)

        with torch.no_grad():
            preds = self._impl(input_tensor)[-1].sigmoid().float().cpu()  # type: ignore[union-attr]

        # Convert back to a PIL grayscale mask at source resolution.
        from torchvision import transforms  # noqa: WPS433  — local import

        pred = preds[0].squeeze()
        mask_low = transforms.ToPILImage()(pred)
        mask = mask_low.resize((orig_w, orig_h), Image.BILINEAR)

        gen_id = uuid.uuid4().hex[:12]

        # Always return the mask itself — useful for downstream
        # compositing pipelines.
        mask_buf = io.BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_b64 = base64.b64encode(mask_buf.getvalue()).decode("ascii")

        result: Dict[str, Any] = {
            "id": gen_id,
            "mask_b64": mask_b64,
            "transparent_b64": None,
            "width": orig_w,
            "height": orig_h,
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }

        if not mask_only:
            cutout = image.copy()
            cutout.putalpha(mask)
            cutout_buf = io.BytesIO()
            cutout.save(cutout_buf, format="PNG")
            result["transparent_b64"] = base64.b64encode(
                cutout_buf.getvalue()
            ).decode("ascii")
            # Also persist to disk so the api's /v1/images/remove-bg
            # response can offer a download URL alongside the inline
            # b64 (matches the /v1/images/3d pattern).
            disk_path = os.path.join(_OUTPUT_DIR, f"{gen_id}.png")
            cutout.save(disk_path)
            result["cutout_path"] = disk_path

        return result

    async def unload(self) -> None:
        if self._impl is not None:
            try:
                self._impl = None
                self._transform = None
                try:
                    import torch  # type: ignore[import-not-found]

                    torch.cuda.empty_cache()
                except Exception:
                    pass
            finally:
                await super().unload()
