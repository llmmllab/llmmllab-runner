# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

llmmllab-runner is a Python 3.12 FastAPI service that dynamically spawns and manages llama.cpp server instances on GPU nodes, then proxies OpenAI-compatible API requests to the correct backend. It runs on Kubernetes with NVIDIA GPU support.

## Development Commands

All commands use `uv` (managed via Makefile). The project requires a `.env` file — copy from `.env.example` and set `LLAMA_SERVER_EXECUTABLE` to your local llama.cpp build path.

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies via `uv sync` |
| `make start` | Run the FastAPI server (port 9000 by default, override with `PORT=8000`) |
| `make validate` | Python syntax check across all `.py` files |
| `make test` | Run pytest (no test suite currently exists) |
| `make clean` | Remove `__pycache__` and `.pyc` files |
| `make docker-build` | Build Docker image (compiles llama.cpp with CUDA in stage 1) |
| `make docker-push` | Push to private registry `192.168.0.71:31500` |
| `make deploy` | Build, push, and deploy to k8s (use `RUNNER_DEPLOY=main\|small\|all`) |
| `make k8s-logs` | Stream runner logs from k8s |
| `make k8s-restart` | Restart runner deployment(s) |

## Architecture

### Request Flow

1. **Gateway** (NGINX Gateway API, in `llmmllab-gateway/` repo) routes `/api/*` to the `llmmllab` Service on port 9999
2. **Runner** receives the request on port 8000
3. **Servers Router** (`routers/servers.py`) — `POST /v1/server/create` acquires or creates a llama.cpp server instance for a model, returning a `server_id` and `base_url`
4. **Proxy Router** (`proxy/router.py`) — catch-all `/v1/server/{server_id}/{path}` rewrites and forwards to the local llama.cpp process on its dynamic port (8001-8900)
5. SSE streaming responses are proxied chunk-by-chunk; client disconnects abort the upstream llama.cpp generation

### Server Lifecycle

- `ServerCache` (`cache.py`) is a thread-safe registry of `ServerEntry` objects with use-count tracking
- Two-tier idle eviction:
  - **Soft** (`CACHE_TIMEOUT_MIN`, default 30min): servers become eligible for eviction when VRAM pressure occurs
  - **Hard** (`EVICTION_TIMEOUT_MIN`, default 60min): servers are forcibly stopped regardless of VRAM
- A background asyncio task runs every 60s calling `evict_idle()` and checking GPU thermals
- `LlamaCppServerManager` (`server_manager/llamacpp.py`) spawns `llama-server` subprocesses, waits for `/health` readiness, and handles graceful shutdown via SIGTERM

### Key Configuration

- Models are defined in `.models.yaml` (YAML list, loaded by `ModelLoader`). Each entry specifies `id`, `details.gguf_file`, `parameters` (num_ctx, n_gpu_layers, etc.), and `pipeline`
- `.env` controls all runtime behavior: timeouts, port ranges, GPU power cap, llama.cpp binary path
- In k8s, models are mounted from hostPath `/models` and the models config is baked into the image or mounted as a volume

### k8s Deployments

- `llmmllab-runner` (main): 3 GPUs, node `lsnode-3`, 8Gi/16Gi memory
- `llmmllab-runner-small`: 1 GPU, node `lsnode-4`, 4Gi/8Gi memory
- Both use `Recreate` strategy (no overlapping deployments) and privileged security context for GPU access
- Both mount hostPath volumes: `/models` (read-only) and `/slots` (read-write for KV cache persistence)
- `SLOT_SAVE_DIR=/slots` is set on both deployments

### Important Implementation Details

- The proxy router increments `use_count` on request start and decrements in a `finally` block when the stream drains or client disconnects — this is what drives the idle eviction timers
- For SSE requests, the upstream connection is closed via `response.aclose()` when the client disconnects, which signals llama.cpp to stop generating
- `BaseServerManager` finds available ports by binding to 127.0.0.1 (intentionally without SO_REUSEADDR to avoid TIME_WAIT collisions)
- Dockerfile is multi-stage: stage 1 compiles llama.cpp from source with CUDA, stage 2 is a runtime-only image with `uv` for dependency management

### Slot Persistence (KV Cache)

When `SLOT_SAVE_DIR` is set, the proxy router (`proxy/router.py`) automatically saves and restores KV cache slots for chat completions with an `X-Session-ID` header:

- **Slot restore** happens before forwarding the request to llama.cpp
- **Slot save** happens in the `finally` block of `upstream_iterator()` (streaming) or after buffering the response (non-streaming), before closing the upstream connection
- **Session-to-slot mapping**: `md5(session_id) % num_slots` — stateless and deterministic
- **Actual slot discovery**: The runner diffs slot token counts before/after each request to find which slot llama.cpp used, since llama.cpp doesn't report it in the response. The mapping is cached per session.
- **Non-streaming requests** are internally proxied as streaming to keep the slot pinned during save, then converted back to JSON
- **Slot cleanup**: A background task in `app.py` periodically deletes old slot files using a session-aware approach: `SLOT_INACTIVE_MAX_AGE_MIN` (default 12 min) checks the proxy's session-activity tracker; `SLOT_CLEANUP_MAX_AGE_MIN` (default 12 min) is an absolute floor for orphaned files; `SLOT_CLEANUP_MAX_SIZE_MB` (default 5000 MB) enforces a directory size cap
- Slot errors must never break the main request flow — all wrapped in try/except

### Session ID Middleware

- `SessionIdMiddleware` in `app.py` extracts `X-Session-ID` header into `_session_id_ctx` context variable
- Session ID flows through structlog entries automatically via `_add_session_id_to_logs` processor
- Accessible in proxy router via `_session_id_ctx.get()`

### Logging Infrastructure

- Loki is exposed through the gateway load balancer at `http://192.168.0.71:3100`
- Query API: `http://192.168.0.71:3100/loki/api/v1/query_range`
- Use `{namespace="llmmllab"}` selector for runner logs
- Logs are stored on a PVC so they survive pod restarts
