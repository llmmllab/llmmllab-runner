# llmmllab-runner

A FastAPI service that dynamically spawns and manages llama.cpp server instances on GPU nodes, proxying OpenAI-compatible API requests to the correct backend. Designed for Kubernetes deployments with NVIDIA GPU support.

## Overview

llmmllab-runner sits between the [llmmllab-api](https://github.com/LongStoryMedia/llmmllab-api) service and llama.cpp server processes. It is the inference backend — llmmllab-api handles authentication, billing, and request orchestration, while the runner manages the actual model loading and GPU execution.

Core responsibilities:

- **Server Lifecycle** — Spawns a new `llama-server` process for each unique model requested, reuses existing instances across clients, and evicts idle servers based on configurable timeouts
- **Request Proxying** — Forwards HTTP requests (including SSE streaming for chat completions) to the correct local llama.cpp instance, aborting upstream generation when clients disconnect
- **GPU Resource Management** — Tracks VRAM usage, applies power caps, monitors thermals, and evicts idle models under VRAM pressure to make room for new ones
- **Model Registry** — Loads model definitions from a YAML config, including GGUF file paths, inference parameters, and LoRA weights

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
| Soft | `CACHE_TIMEOUT_MIN` | 30 min | Server becomes *eligible* for eviction when VRAM pressure occurs |
| Hard | `EVICTION_TIMEOUT_MIN` | 60 min | Server is *forcibly* stopped regardless of VRAM state |

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

### Slot Persistence

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/server/{server_id}/slots/{slot_id}/save` | Save KV cache slot to disk for session persistence |
| `POST` | `/v1/server/{server_id}/slots/{slot_id}/restore` | Restore KV cache slot from disk for session resumption |

Slot persistence eliminates redundant prefill computation by saving the conversation's KV cache to disk between sessions. When enabled (`SLOT_SAVE_DIR` is set), llama-server is launched with `--slot-save-path`, `--no-mmap`, and `--swa-full` flags.

**Workflow:**

1. Set `SLOT_SAVE_DIR=/data/slots` (or any writable directory)
2. After a chat completion, call `POST /v1/server/{server_id}/slots/0/save` to persist the slot
3. On the next session, call `POST /v1/server/{server_id}/slots/0/restore` before sending the next message
4. The restored slot skips re-processing the full prompt history, reducing latency from seconds to milliseconds

Each slot file is named by slot index and lives under `SLOT_SAVE_DIR`. File sizes range from ~1 GB to ~4.4 GB depending on context length.

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with GPU stats and active server count |

## Configuration

All configuration is environment-variable-driven via `.env`. See `config.py` for defaults.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `RUNNER_NAME` | `llmmllab-runner` | Service name (used for tracing identification) |
| `LLAMA_SERVER_EXECUTABLE` | `/llama.cpp/build/bin/llama-server` | Path to the `llama-server` binary |
| `MODELS_FILE_PATH` | *(empty)* | Path to `.models.yaml` — checked before `/app/.models.yaml` |

### Server Lifecycle

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TIMEOUT_MIN` | `30` | Soft eviction timeout (minutes) — server becomes *eligible* for eviction when VRAM pressure occurs |
| `EVICTION_TIMEOUT_MIN` | `60` | Hard eviction timeout (minutes) — server is *forcibly* stopped regardless of VRAM state |
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
| `DCGM_EXPORTER_URL` | `http://localhost:9400/metrics` | DCGM exporter metrics endpoint |
| `DCGM_METRICS_INTERVAL_SEC` | `15` | DCGM metrics scrape interval (seconds) |
| `LLAMA_METRICS_INTERVAL_SEC` | `15` | Llama.cpp server metrics scraping interval (seconds) |

### Task Queue

| Variable | Default | Description |
|----------|---------|-------------|
| `QUEUE_AGING_LOW_TO_MEDIUM_SEC` | `60` | Seconds before a queued task ages from low to medium priority |
| `QUEUE_AGING_MEDIUM_TO_HIGH_SEC` | `120` | Seconds before a queued task ages from medium to high priority |

### Logging & Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `WARNING` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `console` | Log format (`console` for human-readable, `json` for structured) |
| `FORCE_COLOR` | `0` | Force colored output even without TTY (`1` to enable) |
| `TEMPO_ENDPOINT` | `http://tempo.llmmllab.svc.cluster.local:4317` | Jaeger/Tempo OTLP endpoint for distributed tracing |

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
| `SLOT_NO_MMAP` | `true` | Pass `--no-mmap` when `SLOT_SAVE_DIR` is set. Prevents OS from evicting mmap pages between save/restore. |
| `SLOT_SWA_FULL` | `true` | Pass `--swa-full` when `SLOT_SAVE_DIR` is set. Required for SWA models (e.g. Qwen 3.5) to correctly persist their KV cache. |

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
- **`lora_weights`** — Optional list of LoRA adapters to load

## Docker

### Build

```bash
make docker-build
```

The Dockerfile is multi-stage:
1. **Stage 1** — Compiles llama.cpp from source with CUDA support (tag `b8863`)
2. **Stage 2** — Runtime image with the compiled binary, Python 3.12, and `uv` for dependency management

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

## License

Private — see project maintainer for usage terms.

<!-- Test deploy workflow 20260504002035 -->

<!-- trigger deploy -->

