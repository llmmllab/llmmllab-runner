"""
Llama.cpp Argument Builder - Builds command-line arguments for llama.cpp servers.

Builds a config dict from model parameters, then serializes it directly
to a command-line argument list.
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import Model, ModelParameters
from config import (
    LLAMA_SERVER_EXECUTABLE,
    LOG_LEVEL,
    SLOT_SAVE_DIR,
    SLOT_NO_MMAP,
    SLOT_SWA_FULL,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="LlamaCppArgumentBuilder")


class LlamaCppArgumentBuilder:
    """Builds command-line arguments for llama.cpp servers.

    Usage::

        builder = LlamaCppArgumentBuilder(model, port)
        args = builder.build_args()   # ["/llama.cpp/.../llama-server", "--model", "/path/to.gguf", ...]
    """

    def __init__(
        self,
        model: Model,
        port: Optional[int] = None,
        is_embedding: bool = False,
    ):
        self.model = model
        self.port = port
        self.is_embedding = is_embedding

    def build_args(self) -> List[str]:
        """Build the complete argument list for the llama.cpp server process."""
        config = self._build_config()
        args = [LLAMA_SERVER_EXECUTABLE] + _config_to_args(config)
        logger.debug(f"Built args: {' '.join(args)}")
        return args

    def _build_config(self) -> Dict[str, Any]:
        """Build the full config dict for this model."""
        config: Dict[str, Any] = {
            "model": self._get_gguf_path(),
            "host": "127.0.0.1",
            "port": self.port,
        }

        if self.is_embedding:
            self._add_embedding_config(config)
        else:
            self._add_inference_config(config)

        return config

    # --- Embedding ---

    def _add_embedding_config(self, config: Dict[str, Any]) -> None:
        config.update(
            {
                "threads": os.cpu_count() or 4,
                "ctx_size": 4096,
                "batch_size": 1024,
                "embedding": True,
                "pooling": "mean",
                "no_webui": True,
            }
        )
        if LOG_LEVEL.lower() == "debug":
            config["verbose"] = True

    # --- Inference ---

    def _add_inference_config(self, config: Dict[str, Any]) -> None:
        params = self.model.parameters or ModelParameters()

        config.update(
            {
                "cont_batching": True,
                "metrics": True,
                "slots": True,
                "no_warmup": True,
                "flash_attn": "on",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "threads": int(os.cpu_count() or 4),
                "ctx_size": params.num_ctx or 90000,
                "batch_size": params.batch_size or 2048,
                "ubatch_size": params.micro_batch_size or (params.batch_size or 2048),
                "ctx_checkpoints": 24,
                "timeout": 600,
                "context_shift": True,
                "mirostat": 1,
                "cache_ram": 0,
                "parallel": params.parallel or 4,
                "kv_unified": True,
                "repeat_penalty": params.repeat_penalty or 1.1,
                "repeat_last_n": (
                    params.repeat_last_n if params.repeat_last_n is not None else 256
                ),
                "n_gpu_layers": (
                    params.n_gpu_layers if params.n_gpu_layers is not None else -1
                ),
                "main_gpu": params.main_gpu if params.main_gpu is not None else -1,
                "split_mode": params.split_mode or "layer",
                "jinja": True,
                "no_webui": True,
            }
        )

        # Tensor split - only set if configured
        if params.tensor_split:
            config["tensor_split"] = params.tensor_split

        # KV cache placement:
        #   --no-kv-offload tells llama.cpp to keep KV on CPU.
        #   Without it, KV is offloaded to GPU alongside model layers.
        config["no_kv_offload"] = params.kv_on_cpu

        # Multimodal projector
        mmproj_path = self.model.details.clip_model_path
        if mmproj_path and Path(mmproj_path).exists():
            config["mmproj"] = mmproj_path
            logger.info(f"Using multimodal projector: {mmproj_path}")
        elif "vl" in self.model.name.lower() or "vision" in self.model.name.lower():
            logger.warning(
                f"Vision model detected but no mmproj file found for {self.model.name}"
            )

        # Draft model (speculative decoding)
        if hasattr(self.model, "draft_model") and self.model.draft_model:
            if mmproj_path and Path(mmproj_path).exists():
                logger.warning(
                    f"Draft models are not supported with multimodal models. "
                    f"Ignoring draft model for {self.model.name}"
                )
            else:
                from utils.model_loader import ModelLoader

                ml = ModelLoader()
                dm = ml.get_model_by_id(self.model.draft_model)
                draft_gguf = dm.details.gguf_file if dm and dm.details else None
                if draft_gguf:
                    config["model_draft"] = str(draft_gguf)

        # Reasoning (thinking) support
        if params.think:
            config["reasoning"] = "on"
            config["reasoning_budget"] = 16384
            config["reasoning_format"] = "deepseek"
        else:
            config["reasoning"] = "off"

        # Control llama.cpp's --fit auto-reduction: set --fit-ctx so that
        # llama.cpp cannot reduce context below ctx_size_reduction_limit * num_ctx.
        # This prevents silent context shrinkage that breaks conversations.
        ctx_size = params.num_ctx or 90000
        if params.ctx_size_reduction_limit is not None:
            config["fit"] = "on"
            fit_ctx = max(math.ceil(ctx_size * params.ctx_size_reduction_limit), 4096)
            config["fit_ctx"] = fit_ctx
        else:
            config["fit"] = "off"

        # Persistent KV cache slot save/restore for session persistence.
        # llama-server writes slot state to disk under this directory;
        # the REST API (/slots/{id}/save, /slots/{id}/restore) drives
        # the actual save/restore lifecycle at runtime.
        if SLOT_SAVE_DIR:
            save_path = Path(SLOT_SAVE_DIR)
            try:
                save_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.warning(
                    f"Could not create slot save directory {save_path}; "
                    "ensure it exists before starting the server"
                )
            config["slot_save_path"] = str(save_path)
            logger.info(f"Slot persistence enabled: save_path={save_path}")
            # --no-mmap: prevents OS from evicting mmap pages between
            # save and restore, which can corrupt the persisted slot.
            if SLOT_NO_MMAP:
                config["no_mmap"] = True
                logger.info("Slot persistence: --no-mmap enabled")
            # --swa-full: required for SWA models (e.g. Qwen 3.5) to
            # correctly persist their sliding-window KV cache.
            if SLOT_SWA_FULL:
                config["swa_full"] = True
                logger.info("Slot persistence: --swa-full enabled")

        if LOG_LEVEL.lower() == "trace":
            config["verbose"] = True

    # --- Helpers ---

    def _get_gguf_path(self) -> str:
        details = getattr(self.model, "details", None)
        if details and hasattr(details, "gguf_file") and details.gguf_file:
            return details.gguf_file
        return self.model.model


def _config_to_args(config: Dict[str, Any]) -> List[str]:
    """Convert a {flag_name: value} dict to a flat command-line arg list.

    Keys use underscores (python style); they are converted to hyphens
    for the CLI.  Booleans emit a bare flag when True, nothing when False.
    None values are skipped.
    """
    args: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, (list, tuple)):
            if value:
                args.extend([flag, ",".join(map(str, value))])
        else:
            args.extend([flag, str(value)])
    return args
