"""Argument builder for the ``sd-server`` executable.

stable-diffusion.cpp's server binary takes a small, fixed set of CLI flags
that point at model weight files plus a listen-host / listen-port pair.
Per-request generation parameters (steps, cfg_scale, sampler_name, …) are
*not* CLI flags — those travel in the POST body to ``/sdapi/v1/txt2img``.

The builder pulls weight paths from ``Model.details`` (we extended
``ModelDetails`` with ``diffusion_model_path``, ``vae_path``,
``text_encoder_path``, ``text_encoder_kind`` and ``clip_g_path``) and emits
a list ready for ``subprocess.Popen``.
"""

from typing import List, Optional

from config import SD_SERVER_EXECUTABLE
from models import Model
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="SDCppArgumentBuilder")


class SDCppArgumentBuilder:
    """Build the argv list for ``sd-server``."""

    def __init__(self, model: Model, port: Optional[int] = None):
        self.model = model
        self.port = port

    def build_args(self) -> List[str]:
        args: List[str] = [SD_SERVER_EXECUTABLE]

        details = self.model.details

        if details.gguf_file:
            # A single all-in-one .gguf — pass with --model.  Most SDXL/SD15
            # bundles work this way; Qwen-Image splits things up.
            args += ["--model", details.gguf_file]

        if details.diffusion_model_path:
            args += ["--diffusion-model", details.diffusion_model_path]

        if details.vae_path:
            args += ["--vae", details.vae_path]

        if details.text_encoder_path:
            flag_map = {
                "llm": "--llm",
                "clip_l": "--clip_l",
                "t5xxl": "--t5xxl",
            }
            flag = flag_map.get(details.text_encoder_kind or "llm", "--llm")
            args += [flag, details.text_encoder_path]

        if details.clip_g_path:
            args += ["--clip_g", details.clip_g_path]

        # Multi-GPU layout — per-component placement.  sd.cpp does not
        # support llama.cpp-style tensor splitting, but it does let us
        # place clip / diffusion / vae on different backends.
        params = self.model.parameters
        sd_backend = getattr(params, "sd_backend", None) if params else None
        sd_params_backend = (
            getattr(params, "sd_params_backend", None) if params else None
        )
        if sd_backend:
            args += ["--backend", sd_backend]
        if sd_params_backend:
            args += ["--params-backend", sd_params_backend]

        if self.port:
            args += ["--listen-port", str(self.port)]
        args += ["--listen-ip", "127.0.0.1"]

        # Tile the VAE decode so the compute buffer stays small regardless
        # of output resolution.  Without this, decoding a 1024×1024 image
        # with Qwen-Image's WAN VAE allocates a ~7.5 GiB compute buffer on
        # top of the ~18.8 GiB of resident weights — overflows a 24 GiB
        # 3090 and the sampler returns no images with `decode_first_stage
        # failed for latent 1`.  Tile size defaults to 32×32 which is fine
        # for the resolutions we ship.  Negligible quality impact.
        args += ["--vae-tiling"]

        # Verbose by default — sd-server is quiet otherwise and we want logs
        # to flow through our drain threads.
        args += ["-v"]

        logger.debug(f"Built sd-server args: {' '.join(args)}")
        return args
