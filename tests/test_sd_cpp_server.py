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
