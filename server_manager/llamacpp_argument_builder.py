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
        params = self.model.parameters or ModelParameters()
        # llama.cpp requires the whole prompt to fit in one ubatch for
        # embeddings (server.cpp: rejects n_batch > n_ubatch when
        # --embedding is set). Keep batch == ubatch so the server starts.
        batch = params.batch_size or 1024
        config.update(
            {
                "threads": os.cpu_count() or 4,
                "ctx_size": params.num_ctx or 4096,
                "batch_size": batch,
                "ubatch_size": batch,
                "embedding": True,
                "pooling": "mean",
                "no_webui": True,
            }
        )
        # GPU placement: honour the model's params so an embedding model
        # can be pinned to a specific card (e.g. main_gpu: 0 +
        # tensor_split "1,0,0" to keep it off the 3090s reserved for the
        # big chat models). Defaults: offload all layers, let llama.cpp
        # pick the device.
        config["n_gpu_layers"] = (
            params.n_gpu_layers if params.n_gpu_layers is not None else -1
        )
        if params.main_gpu is not None:
            config["main_gpu"] = params.main_gpu
        if params.tensor_split:
            config["tensor_split"] = params.tensor_split
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
                # Honor params.flash_attention (was hardcoded "on", so a yaml
                # flash_attention: False silently did nothing). Default on.
                "flash_attn": (
                    "on"
                    if (params.flash_attention is None or params.flash_attention)
                    else "off"
                ),
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "threads": int(os.cpu_count() or 4),
                "ctx_size": params.num_ctx or 90000,
                "batch_size": params.batch_size or 2048,
                "ubatch_size": params.micro_batch_size or (params.batch_size or 2048),
                # ctx_checkpoints — number of in-memory KV snapshots per slot.
                # Each is ~150 MiB on Qwen3-27B Q6, restored on partial-prefix
                # matches.  llama.cpp default is 8.  We previously hardcoded 32,
                # which on multi-session workloads triggered ~150 MiB restore
                # copies every turn (visible in slot logs).  Default to 8 here;
                # per-model yaml override via ModelParameters.ctx_checkpoints.
                "ctx_checkpoints": (
                    params.ctx_checkpoints if params.ctx_checkpoints is not None else 8
                ),
                "timeout": 600,
                # context_shift — drop oldest tokens on overflow instead of
                # erroring.  Default True (matches llama.cpp default and our
                # historical hardcoded behaviour).
                "context_shift": (
                    params.context_shift if params.context_shift is not None else True
                ),
                # mirostat — adaptive sampler.  Was hardcoded to 1 (Mirostat v1)
                # which adds per-token latency on every generation step and is
                # rarely what you want with top_p/top_k/min_p already configured.
                # Default to 0 (off); per-model yaml can opt back in.
                "mirostat": params.mirostat if params.mirostat is not None else 0,
                # --cache-ram default reduced from 8192 (8 GB) to 2048 (2 GB).
                # The 8 GB default was per-server, and combined with the
                # --no-mmap model load (full 35 GB Q6 in host RAM) was pushing
                # the pod over its memory limit during model switches.
                # 2 GB is more than enough host-side prompt cache for typical
                # workloads; per-model override via ModelParameters.cache_ram
                # is still respected when explicitly set.
                "cache_ram": params.cache_ram if params.cache_ram is not None else 2048,
                "parallel": params.parallel or 4,
                "kv_unified": params.kv_unified,
                "cache_reuse": params.cache_reuse if params.cache_reuse is not None else 256,
                "repeat_penalty": params.repeat_penalty or 1.1,
                "repeat_last_n": (
                    params.repeat_last_n if params.repeat_last_n is not None else 256
                ),
                # Core sampler knobs. These were NOT being passed through any
                # layer (argument builder, the API's ChatOpenAI body, or the
                # proxy's _inject_slot_body), so llama-server silently fell back
                # to its compiled-in defaults (temp 0.80, top_k 40, top_p 0.95,
                # min_p 0.05) — REGARDLESS of .models.yaml. That hot default
                # (temp 0.80) is the fuel for Qwen3.6's word-association
                # cascades. Wiring them here as --temp/--top-k/--top-p/--min-p
                # makes .models.yaml authoritative as the server-side default;
                # a per-request body value still overrides when a caller sends
                # one. None values are dropped by _config_to_args, preserving
                # llama.cpp defaults for any model that omits a knob.
                "temp": params.temperature,
                "top_k": params.top_k,
                "top_p": params.top_p,
                "min_p": params.min_p,
                # presence/frequency penalties (--presence-penalty/--frequency-penalty,
                # fields penalty_present/penalty_freq) — DISTINCT from repeat_penalty
                # (multiplicative) and DRY (n-gram). None-drop keeps llama's 0.0
                # default; left off in yaml for coding models (they can hurt code).
                "presence_penalty": params.presence_penalty,
                "frequency_penalty": params.frequency_penalty,
                # top-n-sigma logit-stddev truncation (--top-n-sigma). None/-1 = off.
                "top_n_sigma": params.top_n_sigma,
                # n_predict (--n-predict) — server-side CAP on generated tokens per
                # response. This IS the OpenAI max_tokens concept (one field). None
                # dropped; -1 = unbounded; a positive int (e.g. 16384) bounds a
                # runaway turn. A per-request max_tokens still overrides this default.
                "n_predict": params.num_predict,
                "dry_multiplier": (
                    params.dry_multiplier if params.dry_multiplier is not None else 0.0
                ),
                "dry_base": (
                    params.dry_base if params.dry_base is not None else 1.75
                ),
                "dry_allowed_length": (
                    params.dry_allowed_length
                    if params.dry_allowed_length is not None
                    else 2
                ),
                "dry_penalty_last_n": (
                    params.dry_penalty_last_n
                    if params.dry_penalty_last_n is not None
                    else 0
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

        # Per-model slot prompt similarity override.
        # Setting to 0 disables LCP-based slot matching, which is required
        # for hash-based slot assignment in session persistence.
        if params.slot_prompt_similarity is not None:
            config["slot_prompt_similarity"] = str(params.slot_prompt_similarity)

        # Seed (--seed). Guarded on >= 0 so the conventional yaml `seed: -1`
        # ("random") does NOT pin a literal seed; only an explicit non-negative
        # seed is emitted. Makes the field truthful instead of silently inert.
        if params.seed is not None and params.seed >= 0:
            config["seed"] = params.seed

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

        # MTP (Multi-Token Prediction) speculative decoding
        if params.spec_type_mtp:
            config["spec_type"] = "draft-mtp"
            config["spec_draft_n_max"] = params.spec_draft_n_max or 16
            logger.info(
                f"MTP speculative decoding enabled: "
                f"spec_draft_n_max={params.spec_draft_n_max or 16}"
            )

        # Reasoning (thinking) support
        if params.think:
            config["reasoning"] = "on"
            config["reasoning_budget"] = params.reasoning_budget or 8192
            config["reasoning_format"] = "deepseek"
            # When the per-token budget is exhausted, llama.cpp injects this
            # text immediately before the closing think tag and then forces
            # the close.  Without it the model is cut off mid-thought with
            # no cue to wrap up, which can produce broken or empty final
            # answers.  See llama.cpp common/arg.cpp::--reasoning-budget-message.
            config["reasoning_budget_message"] = (
                params.reasoning_budget_message
                or "[Thinking budget exhausted — provide your best answer now.]"
            )
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


# Flags that llama.cpp exposes as a paired ``--X`` / ``--no-X`` toggle.
# When the config value is explicitly ``False`` for one of these, we emit
# ``--no-X`` instead of dropping the entry — otherwise yaml ``foo: False``
# silently does nothing because llama.cpp's default for the flag is True.
_NEGATABLE_FLAGS = {
    "context_shift",  # llama.cpp default: enabled
    "kv_unified",     # llama.cpp default: disabled (still useful for explicit-off audit)
}


def _config_to_args(config: Dict[str, Any]) -> List[str]:
    """Convert a {flag_name: value} dict to a flat command-line arg list.

    Keys use underscores (python style); they are converted to hyphens
    for the CLI.  Booleans emit a bare flag when True; for keys in
    :data:`_NEGATABLE_FLAGS`, an explicit ``False`` emits ``--no-X``.
    All other ``False`` booleans and ``None`` values are skipped.
    """
    args: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif key in _NEGATABLE_FLAGS:
                args.append(f"--no-{key.replace('_', '-')}")
        elif isinstance(value, (list, tuple)):
            if value:
                args.extend([flag, ",".join(map(str, value))])
        else:
            args.extend([flag, str(value)])
    return args
