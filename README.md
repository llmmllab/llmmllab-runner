# llmmllab-runner

A FastAPI service that dynamically spawns and manages **llama.cpp** (text) and **stable-diffusion.cpp** (image) server instances on GPU nodes, plus runs an in-process **TRELLIS** pipeline for image-to-3D generation. Proxies OpenAI- and Anthropic-compatible API requests to the right backend. Designed for Kubernetes deployments with NVIDIA GPU support.

## Overview

llmmllab-runner sits between the [llmmllab-api](https://github.com/LongStoryMedia/llmmllab-api) service and the inference backends. It is the inference layer — llmmllab-api handles authentication, billing, and request orchestration, while the runner manages model loading and GPU execution.

Core responsibilities:

- **Server Lifecycle** — Spawns a new `llama-server` (text) or `sd-server` (image) process for each unique model requested, reuses existing instances across clients, and evicts idle servers based on configurable timeouts
- **In-process pipelines** — TRELLIS image-to-3D runs inside the runner process under `/v1/pipelines/img23d/run`; the abstraction at `pipelines/base.py` makes it easy to add more Python-only models (HF diffusers, ComfyUI nodes, etc.)
- **Request Proxying** — Forwards HTTP requests (including SSE streaming for chat completions) to the correct local server, aborting upstream generation when clients disconnect
- **GPU Resource Management** — Tracks VRAM usage, applies power caps, monitors thermals, and evicts idle models under VRAM pressure to make room for new ones
- **Model Registry** — Loads model definitions from a YAML config, including GGUF file paths, inference parameters, and LoRA weights

### Pinned native dependencies (`vendors/`)

Three external sources are vendored as git submodules pinned to specific commits, so the runtime image always matches what's checked into git — no implicit "we pulled master at build time" behaviour:

| Submodule | Used for |
|-----------|----------|
| `vendors/llama.cpp` | Text inference — compiled with CUDA in the Dockerfile, runs as `llama-server` subprocess |
| `vendors/stable-diffusion.cpp` | Image generation + editing — compiled with CUDA, runs as `sd-server` subprocess |
| `vendors/Hunyuan3D-2` | Image-to-3D (`hy3dgen` package) — installed editable in the runtime image, runs in-process |

Update any of them with:

```bash
cd vendors/<name> && git fetch && git checkout <tag-or-commit>
cd ../..
git add vendors/<name>
git commit -m "bump <name> to <tag>"
```

The Makefile target `make vendor-sync` (`git submodule update --init --recursive`) initialises a fresh clone; `make vendor-status` shows the currently-pinned commits. `make vendor-install-py` runs an editable install of the Python-side vendors (currently just `hy3dgen` from `vendors/Hunyuan3D-2`) into the local venv so pyright + IDE jump-to-definition resolve `hy3dgen.*` symbols without needing the Docker image.

### Integration with llmmllab-api

The runner is designed to work with [llmmllab-api](https://github.com/LongStoryMedia/llmmllab-api) as its upstream caller:

- **Shared Pydantic schemas** — The `models/` package contains types identical to `llmmllab-api/models/` so both services share the same request/response contracts
- **Server creation** — llmmllab-api calls `POST /v1/server/create` to acquire a server, then proxies all inference requests through `/v1/server/{server_id}/v1/...`
- **Model config sync** — Both services read from the same `.models.yaml` source, ensuring the API knows which models are available and the runner knows how to load them
- **Gateway routing** — The NGINX Gateway routes `/api/*` to llmmllab-api (port 9999) and `/server/*` directly to the runner (port 8000), allowing the API to coordinate while the runner handles long-lived inference streams

## Architecture

```
Client → NGINX Gateway → llmmllab-runner (:8000)
                              ├── Server Cache (registry of active backends)
                              ├── Proxy Router → llama-server (:8001-:8900)
                              └── GPU Manager (VRAM, thermals, power cap)
```

### Request Flow

1. Client calls `POST /v1/server/create` with a `model_id`
2. Runner checks the `ServerCache` for an existing healthy server for that model
3. If none exists, it spawns a new `llama-server` process on a dynamic port (8001–8900) and waits for `/health` readiness
4. Returns a `server_id` and `base_url` for proxying
5. Subsequent requests go through `POST /v1/server/{server_id}/v1/chat/completions`, which rewrites the path and forwards to the local llama.cpp instance
6. When the client disconnects from a streaming response, the upstream connection is closed, signaling llama.cpp to stop generating

### Two-Tier Idle Eviction

| Tier | Env Var | Default | Behavior |
|------|---------|---------|----------|
| Soft | `CACHE_TIMEOUT_MIN` | 5 min | Server becomes *eligible* for eviction when VRAM pressure occurs. Also drives the api's "paused session vs abandoned server" classifier — see `llmmllab-api` `RunnerClient._select_runner`. |
| Hard | `EVICTION_TIMEOUT_MIN` | 30 min | Server is *forcibly* stopped regardless of VRAM state |

Timers start when the last client releases a server (use_count drops to 0) and reset when a new request arrives.

### Key Components

| Module | Purpose |
|--------|---------|
| `app.py` | FastAPI entry point, lifespan lifecycle, background eviction task |
| `cache.py` | Thread-safe `ServerCache` with use-count tracking and eviction |
| `proxy/router.py` | HTTP proxy with SSE streaming support, client disconnect handling, and slot save/restore endpoints |
| `routers/servers.py` | Server lifecycle: create, status, delete, release, force-evict |
| `routers/models.py` | Model listing and statistics endpoints |
| `server_manager/base.py` | Abstract process manager: spawn, health-check, graceful shutdown |
| `server_manager/llamacpp.py` | llama.cpp-specific manager and argument building |
| `utils/hardware_manager.py` | GPU VRAM tracking, power cap, thermal monitoring |
| `utils/model_loader.py` | Loads model definitions from `.models.yaml` |
| `config.py` | `.env`-based configuration |

## Use Cases

- **Self-hosted LLM inference** — Run multiple models on a single GPU node without manually starting/stopping servers
- **Multi-tenant AI platforms** — Serve different models to different users, with automatic resource reclamation
- **Edge/on-premise deployments** — Deploy to Kubernetes clusters with GPU nodes for low-latency, private inference
- **Model experimentation** — Swap models by updating `.models.yaml` and redeploying, with servers starting on first request

## Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- NVIDIA GPU with CUDA drivers (for GPU-accelerated inference)
- llama.cpp built with CUDA support (for local development)
- Docker (for container builds)
- kubectl (for Kubernetes deployments)

## Quick Start

### 1. Install Dependencies

```bash
make install
```

### 2. Configure

Copy the example environment file and set your llama.cpp binary path:

```bash
cp .env.example .env
```

Edit `.env` — the key setting is `LLAMA_SERVER_EXECUTABLE`:

```
LLAMA_SERVER_EXECUTABLE=/path/to/llama.cpp/build/bin/llama-server
```

### 3. Add Model Definitions

Create `.models.local.yaml` from the example:

```bash
cp .models.example.yaml .models.local.yaml
```

Edit the file to point `details.gguf_file` to your model's GGUF file on disk. The `parameters` section controls inference behavior (context length, GPU layers, temperature, etc.).

### 4. Start the Server

```bash
make start
```

The server starts on port 9000 (override with `PORT=8000 make start`).

### 5. Create a Server Instance

```bash
curl -X POST http://localhost:9000/v1/server/create \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen3_5_0_8B"}'
```

Response:

```json
{
  "server_id": "a1b2c3d4e5f6",
  "base_url": "http://localhost:9000/v1/server/a1b2c3d4e5f6",
  "model": "Qwen3_5_0_8B",
  "port": 8001
}
```

### 6. Send Requests via Proxy

```bash
curl http://localhost:9000/v1/server/a1b2c3d4e5f6/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### 7. Health Check

```bash
curl http://localhost:9000/health
```

Returns GPU stats, active server count, and loaded models.

## API Reference

### Server Lifecycle

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/server/create` | Acquire or create a server for a model |
| `GET` | `/v1/server/{server_id}` | Get server status |
| `DELETE` | `/v1/server/{server_id}` | Stop and remove a server |
| `POST` | `/v1/server/{server_id}/release` | Decrement use count (signal client is done) |
| `POST` | `/v1/server/{server_id}/evict` | Force-evict a server regardless of idle state |

### Proxy

| Method | Path | Description |
|--------|------|-------------|
| `*` | `/v1/server/{server_id}/{path}` | Forward any request to the upstream llama.cpp server |

### Slot Pinning + KV Cache Persistence

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/server/{server_id}/slots/{slot_id}/save` | Save KV cache slot to disk (manual) |
| `POST` | `/v1/server/{server_id}/slots/{slot_id}/restore` | Restore KV cache slot from disk (manual) |

Slot persistence eliminates redundant prefill computation by saving a conversation's KV cache to disk and pinning each session to a stable slot. When enabled, the runner **automatically** manages slot assignment, save-on-evict, and restore-before-use — no manual calls are required.

**How it works:**

1. Set `SLOT_SAVE_DIR` to a writable directory (e.g. `/slots`)
2. Send chat completion requests with an `X-Session-ID` header
3. The runner pins each session to a llama.cpp slot via a per-upstream-server LRU and injects `id_slot`, `cache_prompt=true`, and `n_cache_reuse=256` into the request body so llama.cpp respects the pin and reuses the prefix cache
4. When a session is displaced from its slot by another session, the displaced session's KV state is saved to disk; the incoming session's state is restored if available

**Key details:**

- **Per-server LRU (`SlotLRU`)** — Capacity equals the model's `--parallel`. The first N distinct sessions claim slots `0..N-1`; further sessions evict the least-recently-used session and reuse its slot.
- **Session pinning** — Once assigned, a session stays in the same slot as long as it remains within the LRU window. This makes llama.cpp's slot-local prompt cache hit reliably across turns.
- **KV cache reuse** — The runner passes `--cache-reuse 256` to `llama-server` (configurable per-model via `parameters.cache_reuse` in `.models.yaml`) so llama.cpp can KV-shift to reuse near-prefix matches. The proxy sets the matching `n_cache_reuse` request field.
- **Slot file naming** — `slot_{session_id}_{model_id}.bin` under `SLOT_SAVE_DIR`. Keyed by `model_id` (not the ephemeral `server_id`) so files survive runner restarts and a redeploy doesn't strand the KV state on disk.
- **Pooled httpx clients** — One `httpx.AsyncClient` per upstream server, sized off `--parallel`, with TCP keep-alive. The pool is closed cleanly on shutdown before subprocesses are terminated.
- **Non-streaming requests**: Even non-streaming chat completions are internally proxied as streaming to keep the slot pinned during save, then converted back to JSON for the client.

**Diagnostic: prompt fingerprint logging.** For each chat completion the proxy hashes the request body at offsets `[1K, 2K, 4K, 8K, 12K, 16K, 20K, 24K, 32K, 48K, 64K, 96K, 128K]` and logs one of `FIRST`, `STABLE`, or `DIVERGED at byte N`. This is used to verify the upstream caller is sending byte-stable prompts across turns. Counterpart Prometheus metrics: `prompt_fingerprint_total{kind}` and `prompt_first_divergence_byte`.

**Manual save/restore API:**

For direct control, the proxy exposes explicit save/restore endpoints. Send a JSON body with `{"filename": "slot_name.bin"}`:

```bash
# Save slot 0
curl -X POST http://localhost:9000/v1/server/{server_id}/slots/0/save \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_session.bin"}'

# Restore slot 0
curl -X POST http://localhost:9000/v1/server/{server_id}/slots/0/restore \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_session.bin"}'
```

**Slot file cleanup:**

Slot files grow with conversation length. The runner includes a background cleanup task with three enforcement layers:

1. **Session inactivity** (`SLOT_INACTIVE_MAX_AGE_MIN`, default 12 min): The proxy tracks when each session last had a turn. Slot files for sessions idle longer than this are deleted. Untracked (orphaned) sessions fall back to file mtime.
2. **Absolute age floor** (`SLOT_CLEANUP_MAX_AGE_MIN`, default 12 min): Any slot file older than this is deleted, regardless of session activity. Catches orphaned files where the session wasn't tracked by the proxy.
3. **Size cap** (`SLOT_CLEANUP_MAX_SIZE_MB`, default 5000 MB): Oldest files are trimmed when the directory exceeds this limit.

See [Configuration](#configuration) for details.

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with GPU stats and active server count |
| `GET` | `/v1/status` | `{"startup_epoch": <unix_ms>, "now": <unix_ms>}` — fixed per process; callers detect runner restarts when `startup_epoch` changes and invalidate any cached `server_id` handles |
| `GET` | `/metrics` | Prometheus exposition (see [Metrics](#metrics)) |

### Image generation (stable-diffusion.cpp)

`sd-server` from `vendors/stable-diffusion.cpp` is launched on demand for any model whose `.models.yaml` entry has `provider: stable_diffusion_cpp`. It's reachable through the same proxy:

```bash
# Acquire a server (same flow as llama.cpp)
curl -X POST http://localhost:9000/v1/server/create \
  -H "Content-Type: application/json" \
  -d '{"model_id": "qwen-image-2512"}'

# Send a txt2img request
curl http://localhost:9000/v1/server/<server_id>/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a teacup with steam","width":1024,"height":1024,"steps":40,"cfg_scale":2.5,"sampler_name":"euler"}'
```

The response shape mirrors the AUTOMATIC1111 WebUI: `{"images":["<base64 PNG>"], "parameters": {...}}`. See [Adding a stable-diffusion.cpp model](#adding-a-stable-diffusioncpp-model).

### Pipelines (in-process, e.g. TRELLIS img2-3D)

Pipelines that don't have a standalone server are exposed at `/v1/pipelines/<name>/run`:

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/v1/pipelines` | List registered pipelines and their `loaded` state |
| `POST` | `/v1/pipelines/{name}/run` | Invoke the pipeline; lazy-loads weights on first call |
| `POST` | `/v1/pipelines/{name}/unload` | Release GPU memory |

The TRELLIS pipeline accepts:

```json
{
  "image_b64": "<base64 PNG/JPEG>",
  "seed": 42,
  "ss_steps": 12,
  "slat_steps": 12,
  "formats": ["mesh", "gaussian"]
}
```

and returns paths to the persisted `.glb` mesh and `.ply` gaussian-splat files plus an optional preview frame.

Downloads:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/pipelines/img23d/files/{filename}` | Stream a generated `.glb` / `.ply` / `.png` artefact. Filename must match `<id>.{glb,ply,png}` (alphanumeric id) — anything else is rejected with 400 as a path-traversal guard. Files live under `${TRELLIS_OUTPUT_DIR:-$SD_OUTPUT_DIR/3d}` on the runner. |

The api layer's `GET /v1/images/3d/{filename}` proxies through to this endpoint so clients don't need pod access.

> TRELLIS' CUDA extensions (`gsplat`, custom rasterizer) must be installed in the runner image for the pipeline to load. Without them the endpoint returns HTTP 503 with a structured message naming the missing dependency — the rest of the runner stays functional.

## Configuration

All configuration is environment-variable-driven via `.env`. See `config.py` for defaults.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNER_NAME` | `llmmllab-runner` | Service name (used for tracing identification) |
| `LLAMA_SERVER_EXECUTABLE` | `/llama.cpp/build/bin/llama-server` | Path to the `llama-server` binary |
| `SD_SERVER_EXECUTABLE` | `/stable-diffusion.cpp/build/bin/sd-server` | Path to the `sd-server` binary |
| `SD_OUTPUT_DIR` | `/data/sd-out` | Image-output directory for SD pipelines; sub-pipelines default relative to this |
| `MODELS_FILE_PATH` | *(empty)* | Path to `.models.yaml` — checked before `/app/.models.yaml` |

### Server Lifecycle

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TIMEOUT_MIN` | `10` | Soft eviction timeout (minutes) — server becomes *eligible* for eviction when VRAM pressure occurs. The api mirrors this value to decide when an idle loaded server is safe to commandeer for a new session. |
| `EVICTION_TIMEOUT_MIN` | `30` | Hard eviction timeout in minutes. Cross-repo coupling: the api uses the same value to classify "paused session vs abandoned server".
| `SERVER_START_OOM_RETRIES` | `2` | Max retries when a server start fails due to OOM (set `0` to disable) |

### Network

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNER_PORT` | `8000` | Port the runner listens on |
| `RUNNER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT_RANGE_START` | `8001` | Start of dynamic port range for llama.cpp instances |
| `SERVER_PORT_RANGE_END` | `8900` | End of dynamic port range |
| `PROXY_TIMEOUT` | `600` | Upstream proxy timeout (seconds) — must exceed longest expected inference time |

### GPU & Hardware

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_POWER_CAP_PCT` | `85` | GPU power cap as % of default TDP (`0` to disable) |
| `DCGM_METRICS_ENABLED` | `true` | Enable DCGM exporter metrics scraping (`true`/`false`) |
| `DCGM_EXPORTER_URL` | `http://nvidia-dcgm-exporter.gpu-operator.svc.cluster.local:9400/metrics` | DCGM exporter metrics endpoint (Kubernetes ClusterIP across GPU nodes; filter by `NODE_NAME`) |
| `DCGM_METRICS_INTERVAL_SEC` | `15` | DCGM metrics scrape interval (seconds) |
| `LLAMA_METRICS_INTERVAL_SEC` | `15` | Llama.cpp server metrics scraping interval (seconds) |
| `NODE_NAME` | *(empty)* | Kubernetes node name (set via downward API); used to filter DCGM metrics to the local node |

### Logging & Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `WARNING` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `console` | Log format (`console` for human-readable, `json` for structured) |
| `FORCE_COLOR` | `0` | Force colored output even without TTY (`1` to enable) |
| `TEMPO_ENDPOINT` | `http://tempo.llmmllab.svc.cluster.local:4317` | Jaeger/Tempo OTLP endpoint for distributed tracing |

### Pipeline Configuration

These variables control the image-generation sub-pipelines (img23d, mesh2parts, rembg):

| Variable | Default | Description |
|----------|---------|-------------|
| `HUNYUAN3D_MODEL_PATH` | `tencent/Hunyuan3D-2.1` | HuggingFace model path for img23d pipeline |
| `HUNYUAN3D_SUBFOLDER` | `hunyuan3d-dit-v2-1` | Model subfolder within the HuggingFace repo |
| `HUNYUAN3D_USE_SAFETENSORS` | `false` | Use safetensors format for model loading (`true`/`false`) |
| `TRELLIS_OUTPUT_DIR` | `$SD_OUTPUT_DIR/3d` | Output directory for img23d (Trellis) pipeline |
| `HUNYUAN3D_PART_OUTPUT_DIR` | `$SD_OUTPUT_DIR/3d_parts` | Output directory for mesh2parts pipeline |
| `HUNYUAN3D_PART_OCTREE_RESOLUTION` | `512` | Octree resolution for mesh2parts |
| `HUNYUAN3D_PART_MAX_PARTS` | `0` | Maximum parts for mesh2parts (`0` = unlimited) |
| `HUNYUAN3D_PART_DTYPE` | `fp16` | Data type for mesh2parts computation (`fp16`, `bf16`, `fp32`) |
| `RMBG_OUTPUT_DIR` | `$SD_OUTPUT_DIR/rembg` | Output directory for rembg pipeline |
| `RMBG_INPUT_SIZE` | `1024` | Input size for rembg pipeline |
| `IN_PROCESS_AUTO_UNLOAD` | *(empty)* | If truthy, auto-unload pipeline models after use to reclaim VRAM |

### Runtime (GPU / Allocator)

These are set in the Kubernetes deployment and affect the underlying runtime:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | *(unset)* | GPU device IDs visible to the process |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | CUDA device ordering |
| `CUDA_LAUNCH_BLOCKING` | `0` | Synchronous CUDA error checking (`1` for debugging) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | NVIDIA container runtime device visibility |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video` | NVIDIA driver capabilities |
| `PYTHONMALLOC` | `malloc` | Python memory allocator |
| `MALLOC_ARENA_MAX` | `2` | glibc malloc arena limit (reduces memory fragmentation) |
| `GGML_LOG_LEVEL` | `2` | GGML (llama.cpp backend) log verbosity |
| `SLOT_SAVE_DIR` | (empty) | Directory for persistent KV cache slots. When set, passes `--slot-save-path` to llama-server, enabling session state persistence via the slot save/restore API. See [Slot Persistence](#slot-persistence) below. |
| `SLOT_NO_MMAP` | `false` | Pass `--no-mmap` when `SLOT_SAVE_DIR` is set (opt-in). Prevents OS from evicting mmap pages between save/restore. Set to `true` only if you hit specific page-eviction issues with saved slots.
| `SLOT_SWA_FULL` | `true` | Pass `--swa-full` when `SLOT_SAVE_DIR` is set. Required for SWA models (e.g. Qwen 3.5) to correctly persist their KV cache. |
| `SLOT_INACTIVE_MAX_AGE_MIN` | `12` | Delete slot files for sessions inactive for this many minutes. Uses the proxy's session-activity tracker. Set to `0` to disable. |
| `SLOT_CLEANUP_MAX_AGE_MIN` | `12` | Absolute age floor: delete any slot file older than this many minutes, regardless of session activity. Catches orphaned files. Set to `0` to disable. |
| `SLOT_CLEANUP_MAX_SIZE_MB` | `5000` | Maximum total size of slot directory in MB. Oldest files are deleted when exceeded. Set to `0` to disable. |
| `SLOT_CLEANUP_INTERVAL_SEC` | `300` | How often the slot cleanup task runs. |

## Model Configuration

Models are defined in a YAML file (default `.models.yaml` or path from `MODELS_FILE_PATH`). Each entry:

```yaml
- id: Qwen3_5_0_8B
  name: "Qwen3.5-0.8B"
  model: "Qwen3.5-0.8B"
  task: TextToText
  details:
    format: gguf
    gguf_file: "/models/qwen3.5/0.8b/q4_k_m.gguf"
    family: qwen
    parameter_size: 752.4M
    quantization_level: Q4_K_M
    precision: int4
    size: 771092736
    original_ctx: 2048
  parameters:
    num_ctx: 50000
    n_gpu_layers: -1
    batch_size: 2048
    micro_batch_size: 1024
    temperature: 0.7
    top_p: 0.95
    min_p: 0.05
  pipeline: llama
  provider: llama_cpp
```

Key fields:
- **`id`** — Unique identifier used in `/v1/server/create` requests
- **`details.gguf_file`** — Absolute path to the GGUF model file on disk
- **`parameters`** — Passed as command-line args to `llama-server` (num_ctx, n_gpu_layers, batch_size, etc.)
- **`parameters.cache_reuse`** — Minimum chunk size for KV-shift prefix reuse (`--cache-reuse`). Defaults to `256` when unset. Required for per-session prompt-cache reuse to take effect.
- **`parameters.split_mode`** — `layer` or `row`. Use `layer` if `--parallel > 1`: `row` triggers `GGML_ASSERT(!(split && ne02 < ne12))` in `ggml-cuda.cu` when two slots pack into the same ubatch and takes the whole `llama-server` process down.
- **`parameters.micro_batch_size`** — Keep meaningfully smaller than `batch_size` (e.g. `512` vs `2048`) when running multiple slots so llama.cpp can interleave tokens from multiple slots inside each micro-batch instead of letting one slot monopolize prefill.
- **`lora_weights`** — Optional list of LoRA adapters to load

### Adding a stable-diffusion.cpp model

Image models use the same YAML file but populate a different set of fields:

```yaml
- id: qwen-image-2512
  name: "Qwen-Image-2512 Q4_K_M"
  model: "qwen-image-2512"
  task: TextToImage
  modified_at: "2026-05-23T00:00:00+00:00"
  digest: "0000000000000000000000000000000000000000000000000000000000000000"
  details:
    format: gguf
    family: qwen-image
    families: [qwen-image]
    parameter_size: 20B
    quantization_level: Q4_K_M
    specialization: TextToImage
    size: 12000000000
    original_ctx: 0           # SD has no LLM-style context window
    diffusion_model_path: /models/qwen-image/qwen-image-2512-Q4_K_M.gguf
    vae_path: /models/qwen-image/qwen_image_vae.safetensors
    text_encoder_path: /models/qwen-image/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
    text_encoder_kind: llm    # llm | clip_l | t5xxl
  provider: stable_diffusion_cpp
```

The runner picks `SDCppServerManager` when it sees `provider: stable_diffusion_cpp` and translates the `details.*` paths to the matching `sd-server` flags (`--diffusion-model`, `--vae`, `--llm`/`--clip_l`/`--t5xxl`, `--clip_g`). For SDXL-style all-in-one GGUFs, set `details.gguf_file` and leave the split-file fields unset.

#### Multi-GPU layouts for SD

stable-diffusion.cpp does **not** support llama.cpp-style tensor splitting (slicing one layer's weights across devices), but it does let you place each component (text encoder, diffusion, VAE) on a different backend. Two `parameters` fields expose this:

| Field | sd-server flag | Purpose |
|-------|----------------|---------|
| `sd_backend` | `--backend` | Per-component **compute** placement |
| `sd_params_backend` | `--params-backend` | Per-component **weight storage** placement |

Examples:

```yaml
# Spread Qwen-Image across two 3090s: text encoder on cuda0, diffusion+VAE on cuda1.
# Removes the need for --vae-tiling at 1024×1024 and frees a card for the OTHER model.
parameters:
  sd_backend: "clip=cuda0,diffusion=cuda1,vae=cuda1"
```

```yaml
# Trade ~30% throughput for VRAM headroom: weights live in RAM, compute on GPU.
parameters:
  sd_params_backend: "diffusion=cpu"
```

When `sd_backend` is set, the runner intentionally does **not** narrow `CUDA_VISIBLE_DEVICES` (otherwise the `cuda1`/`cuda2` names in your layout string wouldn't resolve). `main_gpu` is ignored in that case.

The simpler single-GPU pin (`main_gpu: 2`) is still the right choice when one card has enough VRAM — we use it by default for Qwen-Image-2512 (pinned to a 24 GB 3090).

## Metrics

The runner exposes Prometheus metrics at `/metrics`. Cardinality is intentionally bounded — no `session_id` labels.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `slot_resolutions_total` | counter | `slot_id`, `evicted` | One per `SlotLRU.touch()`; `evicted=true` when the touch displaced an existing session |
| `slot_lru_size` | gauge | `server_id` | Current SlotLRU population per upstream server |
| `slot_save_total` | counter | `slot_id`, `outcome` | Save-on-evict outcomes (`success` / `failure`) |
| `slot_restore_total` | counter | `slot_id`, `outcome` | Restore-before-use outcomes |
| `slot_save_duration_seconds` | histogram | — | Wall-clock of the upstream `/slots/{id}?action=save` call |
| `slot_restore_duration_seconds` | histogram | — | Wall-clock of the upstream `/slots/{id}?action=restore` call |
| `prompt_fingerprint_total` | counter | `kind` (`first`/`stable`/`diverged`) | Prompt prefix stability across turns of a session |
| `prompt_first_divergence_byte` | histogram | — | Offset of first divergence; lower buckets = worse cache reuse |
| `prompt_body_bytes` | histogram | — | Request body size distribution |
| `llama_server_evictions_total` | counter | `reason` | `reason="process_died"` fires from the per-subprocess watchdog on unexpected exit |

## Resilience

### Process-death watchdog

Each llama.cpp subprocess has a watchdog thread (`server_manager/base.py::_watchdog_run`) that waits on the process. On unexpected exit — GGML_ASSERT, SIGKILL, segfault, OOM — the watchdog:

1. Snapshots the last 50 lines from a 200-line stderr ring buffer (drained at INFO so live CUDA errors surface).
2. Logs a WARNING with the stderr tail and exit code.
3. Calls `ServerCache.purge_dead_server(server_id)` to drop the cache entry immediately, so the proxy stops routing to the dead port instead of 502-storming until the 30-60 minute TTL eviction.
4. Bumps `llama_server_evictions_total{reason="process_died"}`.

An `_intentional_stop` flag, set by `stop()`, the reduced-context retry path, and the cache eviction path, distinguishes "we asked the process to exit" from "it died." The watchdog is a no-op for intentional stops.

### Shutdown ordering

`app.py` lifespan tears down in this order on shutdown:

1. Cancel background tasks (eviction, slot cleanup, metrics, etc.) so nothing borrows new connections.
2. `aclose_all_clients()` — close the pooled per-upstream-server `httpx.AsyncClient` instances so TCP keep-alive connections drain cleanly.
3. `server_cache.stop_all()` — terminate llama.cpp subprocesses last.

### Restart resilience

- **Slot files survive restarts** — Slot save files are keyed by `model_id`, not the ephemeral per-subprocess `server_id`. A pod restart that re-launches the same model picks up the previous KV state from disk.
- **api-side handle invalidation** — `GET /v1/status` returns a `startup_epoch` captured at module import. The api polls this; when it changes, it invalidates cached `server_id` handles and reacquires them transparently.

## Docker

### Build

```bash
make docker-build
```

The Dockerfile is multi-stage:
1. **Stage 1 — `llama-builder`** — Compiles `llama.cpp` from `vendors/llama.cpp` with CUDA support
2. **Stage 2 — `sd-builder`** — Compiles `stable-diffusion.cpp` from `vendors/stable-diffusion.cpp` with CUDA support
3. **Stage 3 — runtime** — Copies the compiled `llama-server` and `sd-server` binaries plus their shared libraries into a CUDA 12.8 runtime image, then layers in Python 3.12 + `uv` and the runner's source

`make docker-build` automatically runs `make vendor-sync` first to make sure the submodules are initialised before COPY.

### Run Locally

```bash
make docker-run
```

Runs with `--gpus all` and mounts `.models.yaml` into the container.

### Push to Registry

```bash
make docker-push
```

Pushes to `192.168.0.71:31500` (override with `REGISTRY=` and `TAG=`).

### Registry retention / cleanup

CI pushes a `:${SHA}`-tagged image on every commit; the LAN registry fills up fast (the api repo alone had 240 stale tags = ~72 GiB at the first sweep).

`scripts/registry_cleanup.py` keeps the N most recent tags per repo and deletes the rest via the Distribution v2 API. Auth is read from the existing `registry-secret` (dockerconfigjson) — no separate credential to maintain.

```bash
# Dry-run against the live registry (no deletes; shows the plan)
REGISTRY_USER=... REGISTRY_PASSWORD=... \
    python3 scripts/registry_cleanup.py -v

# Or via the existing dockerconfigjson secret
DOCKER_CONFIG_JSON="$(kubectl get secret -n llmmllab registry-secret \
    -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d)" \
    python3 scripts/registry_cleanup.py -v

# Apply
python3 scripts/registry_cleanup.py --apply --keep=5
```

Schedule it nightly via the included CronJob — runs as a small `python:3.12-slim` pod with the script mounted from a ConfigMap and the registry credentials mounted from the `registry-secret`:

```bash
# Refresh the ConfigMap from the source script + apply the CronJob
kubectl create configmap registry-cleanup-script \
    -n llmmllab \
    --from-file=registry_cleanup.py=scripts/registry_cleanup.py \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/registry-cleanup-cronjob.yaml
```

**Reclaiming the disk** — `DELETE /v2/<repo>/manifests/<digest>` only dereferences the layers; the registry still holds the blobs until garbage-collect runs against the binary itself. Once the cron has trimmed manifests for a while, run GC on whatever node runs the registry container:

```bash
# Example for a `registry:2` docker container running on the host:
docker exec <registry-container> \
    registry garbage-collect /etc/docker/registry/config.yml -m

# Or, if the registry binary is run directly:
sudo registry garbage-collect /etc/docker/registry/config.yml -m
```

The `-m` flag drops the deleted-manifest stubs as well; without it you only reclaim layer space, not the per-tag metadata.

## Kubernetes Deployment

### Deploy

```bash
# Deploy to main runner (3 GPUs, lsnode-3)
make deploy

# Deploy to small runner (1 GPU, lsnode-4)
RUNNER_DEPLOY=small make deploy

# Deploy to both
RUNNER_DEPLOY=all make deploy
```

### Common Operations

```bash
# Stream logs
make k8s-logs

# Restart deployment
make k8s-restart

# Restart small runner only
RUNNER_DEPLOY=small make k8s-restart
```

### Deployment Topology

| Deployment | Node | GPUs | Memory | Use Case |
|------------|------|------|--------|----------|
| `llmmllab-runner` | lsnode-3 | 3 (CUDA 0,1,2) | 8Gi/16Gi | Large models |
| `llmmllab-runner-small` | lsnode-4 | 1 (CUDA 0) | 4Gi/8Gi | Small models / high concurrency |

Both use `Recreate` strategy to prevent overlapping instances on the same node.

### Persistent Volumes

Both deployments mount hostPath volumes for slot persistence:

| Volume | Host Path | Container Mount | Purpose |
|--------|-----------|-----------------|---------|
| `models` | `/models` | `/models` (read-only) | Model GGUF files |
| `slots` | `/slots` | `/slots` (read-write) | KV cache slot files |

The `SLOT_SAVE_DIR` environment variable is set to `/slots` on both deployments. Slot files persist across pod restarts and are cleaned up automatically by the background cleanup task. Ensure the host directories exist on the target nodes (`lsnode-3` and `lsnode-4`).

## Development

### Project Structure

```
llmmllab-runner/
├── app.py                      # FastAPI app, lifespan, health endpoint
├── cache.py                    # ServerCache with two-tier eviction
├── config.py                   # .env-based configuration
├── pyproject.toml              # Dependencies (uv-managed)
├── Makefile                    # Build, run, deploy targets
├── Dockerfile                  # Multi-stage: compile llama.cpp + runtime
├── proxy/
│   └── router.py               # HTTP proxy with SSE streaming
├── routers/
│   ├── servers.py              # Server lifecycle endpoints
│   └── models.py               # Model listing endpoints
├── server_manager/
│   ├── base.py                 # Abstract process manager
│   ├── llamacpp.py             # llama.cpp server manager
│   └── llamacpp_argument_builder.py  # CLI arg construction
├── utils/
│   ├── hardware_manager.py     # GPU VRAM, thermals, power cap
│   ├── model_loader.py         # YAML model config loader
│   └── logging.py              # Structured logging with structlog
├── models/                     # Pydantic schemas (shared with llmmllab-api)
├── k8s/
│   ├── deployment.yaml         # k8s Deployment manifests
│   ├── service.yaml            # k8s Service
│   ├── referencegrant.yaml     # Gateway API ReferenceGrant
│   ├── pvc.yaml                # PersistentVolumeClaim
│   └── apply.sh                # Deploy script
├── .env.example                # Environment variable template
├── .models.example.yaml        # Model definition template
└── .models.yaml                # Active model config (git-ignored)
```

### Running Locally

```bash
# Install dependencies
make install

# Copy and configure .env
cp .env.example .env
# Edit .env — set LLAMA_SERVER_EXECUTABLE to your local build

# Copy and configure models
cp .models.example.yaml .models.local.yaml
# Edit .models.local.yaml — set gguf_file paths

# Start the server
make start
```

### Code Quality

```bash
# Syntax check all Python files
make validate

# Clean cache files
make clean
```

### Debugging

Set `LOG_LEVEL=DEBUG` in `.env` to see:
- Full llama.cpp command-line arguments for each server start
- Subprocess stdout/stderr streamed into the runner logs
- Health check polling during server startup
- Proxy request/response details

The `/health` endpoint is useful for runtime diagnostics:

```bash
curl http://localhost:9000/health | python3 -m json.tool
```

Returns per-GPU memory usage, active server list with ports and use counts, and all loaded models.

## Contributing

### Before Submitting Changes

1. Run `make validate` to ensure syntax is correct
2. Test locally with a real llama.cpp build and at least one model
3. Verify the `/health` endpoint shows expected GPU and server state
4. For proxy changes, test both streaming and non-streaming request paths
5. For eviction changes, verify with `CACHE_TIMEOUT_MIN=1` and `EVICTION_TIMEOUT_MIN=1` in `.env`

### Key Areas to Understand

- **Use count tracking** — The proxy router increments on request start and decrements in a `finally` block. This is the single source of truth for idle state. If you change how requests flow, ensure use_count stays accurate.
- **SSE streaming** — The proxy detects streaming requests by checking the `stream` field in the JSON body. If the client disconnects, `response.aclose()` closes the upstream TCP connection, which aborts llama.cpp generation.
- **Port allocation** — `BaseServerManager._find_available_port()` binds to 127.0.0.1 without `SO_REUSEADDR` to avoid TIME_WAIT collisions. Ports are allocated sequentially from 8001.
- **VRAM eviction** — When a new model is requested and VRAM is insufficient, the soft eviction path stops idle servers oldest-first until enough memory is free. The model size estimate is `details.size + 128MB` overhead.
- **Slot pinning + persistence** — `proxy/router.py` pins each `X-Session-ID` to a llama.cpp slot via a per-upstream-server `SlotLRU` (capacity = `--parallel`). Before forwarding, the proxy injects `id_slot`, `cache_prompt=true`, and `n_cache_reuse=256` into the upstream JSON body. When a session is evicted from its slot to make room for another, the proxy saves the displaced KV state to disk and (on the next forward for the incoming session) restores from disk if available. Slot files are keyed by `model_id` so they survive runner restarts. Slot errors must never break the main request flow — all operations are wrapped in try/except.
- **Session ID middleware** — `SessionIdMiddleware` in `app.py` extracts the `X-Session-ID` header into a context variable (`_session_id_ctx`), making it available throughout the request lifecycle, including in structlog entries.

## License

Private — see project maintainer for usage terms.

<!-- Test deploy workflow 20260504002035 -->

<!-- trigger deploy -->

