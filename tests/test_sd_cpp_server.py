"""Unit tests for the stable-diffusion.cpp server manager + arg builder.

The SD server runs as a subprocess just like llama.cpp; these tests exercise
the boundary code (CLI flag generation, endpoint mapping) without actually
spawning the binary.
"""

from models import Model, ModelDetails, ModelProvider, ModelTask
from server_manager import SDCppArgumentBuilder, SDCppServerManager


def _make_sd_model(**details_overrides) -> Model:
    """Build a minimal SD-flavoured ``Model`` for the tests."""
    base_details = {
        "format": "gguf",
        "family": "qwen-image",
        "families": ["qwen-image"],
        "parameter_size": "20B",
        "size": 1,
        "original_ctx": 0,
        "specialization": "TextToImage",
        "diffusion_model_path": "/models/qwen-image/qwen-image-2512-Q4_K_M.gguf",
        "vae_path": "/models/qwen-image/qwen_image_vae.safetensors",
        "text_encoder_path": "/models/qwen-image/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf",
        "text_encoder_kind": "llm",
    }
    base_details.update(details_overrides)
    return Model(
        id="qwen-image",
        name="Qwen-Image-2512",
        model="qwen-image",
        task=ModelTask.TEXTTOIMAGE,
        modified_at="2026-05-23T00:00:00+00:00",
        digest="0" * 64,
        details=ModelDetails(**base_details),
        provider=ModelProvider.STABLE_DIFFUSION_CPP,
    )


def test_arg_builder_emits_qwen_image_layout():
    """The Qwen-Image GGUF split layout must produce all four model flags."""
    model = _make_sd_model()
    args = SDCppArgumentBuilder(model, port=8500).build_args()

    assert "--diffusion-model" in args
    assert args[args.index("--diffusion-model") + 1].endswith("qwen-image-2512-Q4_K_M.gguf")
    assert "--vae" in args
    assert args[args.index("--vae") + 1].endswith("qwen_image_vae.safetensors")
    # text_encoder_kind=llm -> --llm
    assert "--llm" in args
    assert args[args.index("--llm") + 1].endswith("Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf")
    assert "--listen-port" in args
    assert args[args.index("--listen-port") + 1] == "8500"


def test_arg_builder_routes_clip_l_and_t5xxl_text_encoders():
    """text_encoder_kind switches between --llm / --clip_l / --t5xxl."""
    clip_model = _make_sd_model(text_encoder_kind="clip_l")
    t5_model = _make_sd_model(text_encoder_kind="t5xxl")

    clip_args = SDCppArgumentBuilder(clip_model).build_args()
    t5_args = SDCppArgumentBuilder(t5_model).build_args()

    assert "--clip_l" in clip_args
    assert "--llm" not in clip_args
    assert "--t5xxl" in t5_args
    assert "--llm" not in t5_args


def test_arg_builder_handles_single_file_model():
    """SDXL-style all-in-one .gguf goes through --model instead of --diffusion-model."""
    model = _make_sd_model(
        diffusion_model_path=None,
        vae_path=None,
        text_encoder_path=None,
        gguf_file="/models/sdxl/sdxl-base.gguf",
    )
    args = SDCppArgumentBuilder(model).build_args()

    assert "--model" in args
    assert args[args.index("--model") + 1] == "/models/sdxl/sdxl-base.gguf"
    assert "--diffusion-model" not in args
    assert "--llm" not in args


def test_arg_builder_emits_clip_g_when_set():
    """clip_g_path adds --clip_g (used by SDXL/SD3)."""
    model = _make_sd_model(clip_g_path="/models/clip_g.safetensors")
    args = SDCppArgumentBuilder(model).build_args()

    assert "--clip_g" in args
    assert args[args.index("--clip_g") + 1] == "/models/clip_g.safetensors"


def test_arg_builder_always_enables_vae_tiling():
    """The Qwen-Image WAN VAE blows past 24 GiB on a 1024 decode without
    ``--vae-tiling``.  We always pass it; the quality hit is negligible
    and the alternative is a hard OOM mid-generation."""
    model = _make_sd_model()
    args = SDCppArgumentBuilder(model).build_args()
    assert "--vae-tiling" in args


def test_arg_builder_passes_multi_gpu_backend():
    """``sd_backend`` and ``sd_params_backend`` propagate to the sd-server
    CLI so a yaml entry can lay components across multiple GPUs."""
    from models import ModelParameters

    model = _make_sd_model()
    model.parameters = ModelParameters(
        sd_backend="clip=cuda0,diffusion=cuda1,vae=cuda1",
        sd_params_backend="diffusion=cpu",
    )
    args = SDCppArgumentBuilder(model).build_args()

    assert "--backend" in args
    assert args[args.index("--backend") + 1] == "clip=cuda0,diffusion=cuda1,vae=cuda1"
    assert "--params-backend" in args
    assert args[args.index("--params-backend") + 1] == "diffusion=cpu"


def test_sd_server_manager_skips_single_gpu_pin_when_multi_gpu_layout_set():
    """When ``sd_backend`` references multiple devices, pinning
    CUDA_VISIBLE_DEVICES to a single one would break the layout — the
    env hook must inherit instead."""
    from models import ModelParameters

    model = _make_sd_model()
    model.parameters = ModelParameters(
        main_gpu=1,
        sd_backend="clip=cuda0,diffusion=cuda1",
    )
    mgr = SDCppServerManager(model=model, port=9999)
    # Even though main_gpu=1, multi-GPU layout takes precedence.
    assert mgr._build_subprocess_env() is None


def test_sd_server_manager_health_maps_to_capabilities():
    """``/health`` rewrites to the only endpoint that proves the model is loaded."""
    model = _make_sd_model()
    mgr = SDCppServerManager(model=model, port=9999)
    assert mgr.get_api_endpoint("/health").endswith("/sdcpp/v1/capabilities")


def test_sd_server_manager_passthrough_paths():
    """Non-health paths are forwarded verbatim (no /v1 prefix)."""
    model = _make_sd_model()
    mgr = SDCppServerManager(model=model, port=9999)
    assert mgr.get_api_endpoint("/sdapi/v1/txt2img").endswith("/sdapi/v1/txt2img")
    assert mgr.get_api_endpoint("/v1/images/generations").endswith("/v1/images/generations")


def test_sd_server_manager_skips_context_validation():
    """SD has no context; the validator must short-circuit to True."""
    model = _make_sd_model()
    mgr = SDCppServerManager(model=model, port=9999)
    assert mgr._validate_context_size() is True


def test_sd_server_manager_pins_cuda_visible_devices_when_main_gpu_set():
    """``main_gpu`` on the model parameters must propagate to
    ``CUDA_VISIBLE_DEVICES`` on the child process — otherwise sd-server
    defaults to device 0 and crashes with OOM on smaller cards."""
    import os
    from models import ModelParameters

    model = _make_sd_model()
    model.parameters = ModelParameters(main_gpu=2)
    mgr = SDCppServerManager(model=model, port=9999)

    env = mgr._build_subprocess_env()
    assert env is not None, "Env must be set when main_gpu >= 0"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    # Other env vars should still be there (it's os.environ.copy() + override).
    assert "PATH" in env or os.environ.get("PATH") is None


def test_sd_server_manager_inherits_env_when_main_gpu_unset():
    """``main_gpu = -1`` (the default) should NOT touch the env — child
    inherits whatever the runner has."""
    model = _make_sd_model()
    # main_gpu defaults to -1 on ModelParameters
    mgr = SDCppServerManager(model=model, port=9999)

    env = mgr._build_subprocess_env()
    assert env is None


def test_sd_server_manager_inherits_env_when_parameters_missing():
    """No ``parameters`` block at all should also be a no-op."""
    model = _make_sd_model()
    model.parameters = None
    mgr = SDCppServerManager(model=model, port=9999)

    assert mgr._build_subprocess_env() is None


# ---------------------------------------------------------------------------
# Qwen-Image-Edit-specific flags
# ---------------------------------------------------------------------------


def test_arg_builder_emits_llm_vision_when_set():
    """The Qwen-Image-Edit 2509+ instruction-following path requires the
    Qwen2.5-VL visual tower; ``llm_vision_path`` must propagate to the
    sd-server ``--llm_vision`` flag."""
    model = _make_sd_model(
        llm_vision_path="/models/qwen2.5-vl/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    )
    args = SDCppArgumentBuilder(model).build_args()

    assert "--llm_vision" in args
    assert args[args.index("--llm_vision") + 1].endswith(
        "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    )


def test_arg_builder_omits_llm_vision_when_unset():
    """No llm_vision_path → no flag.  Plain txt2img models don't need it."""
    model = _make_sd_model()  # no llm_vision_path
    args = SDCppArgumentBuilder(model).build_args()
    assert "--llm_vision" not in args


def test_arg_builder_emits_qwen_image_zero_cond_t():
    """Required for Qwen-Image-Edit-2511; without it editing quality
    degrades significantly per leejet/stable-diffusion.cpp docs."""
    from models import ModelParameters

    model = _make_sd_model()
    model.parameters = ModelParameters(qwen_image_zero_cond_t=True)
    args = SDCppArgumentBuilder(model).build_args()

    assert "--qwen-image-zero-cond-t" in args


def test_arg_builder_omits_zero_cond_when_false_or_unset():
    """Boolean flag — omit unless explicitly true."""
    from models import ModelParameters

    for params in (None, ModelParameters(), ModelParameters(qwen_image_zero_cond_t=False)):
        model = _make_sd_model()
        model.parameters = params
        args = SDCppArgumentBuilder(model).build_args()
        assert "--qwen-image-zero-cond-t" not in args, f"unexpected zero_cond_t with params={params!r}"


def test_arg_builder_emits_flow_shift():
    from models import ModelParameters

    model = _make_sd_model()
    model.parameters = ModelParameters(flow_shift=3.0)
    args = SDCppArgumentBuilder(model).build_args()

    assert "--flow-shift" in args
    assert args[args.index("--flow-shift") + 1] == "3.0"
