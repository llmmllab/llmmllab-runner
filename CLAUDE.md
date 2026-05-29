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
  - **Soft** (`CACHE_TIMEOUT_MIN`, default 5min): servers become eligible for eviction when VRAM pressure occurs. Also the threshold the api uses to distinguish "session paused mid-conversation, don't preempt" from "abandoned, safe to commandeer for a new session."
  - **Hard** (`EVICTION_TIMEOUT_MIN`, default 30min): servers are forcibly stopped regardless of VRAM
- A background asyncio task runs every 60s calling `evict_idle()` and checking GPU thermals
- `LlamaCppServerManager` (`server_manager/llamacpp.py`) spawns `llama-server` subprocesses, waits for `/health` readiness, and handles graceful shutdown via SIGTERM
- A per-manager watchdog thread purges crashed subprocesses from `ServerCache` immediately (see "Process-Death Watchdog" below) so the proxy stops routing to a dead port rather than 502-storming until TTL eviction

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

### Slot Pinning + KV Cache Persistence

The proxy (`proxy/router.py`) pins each `X-Session-ID` to a specific llama.cpp slot via a per-upstream-server LRU, eagerly injects slot/cache parameters into the upstream JSON body, and saves/restores KV state on eviction so sessions survive both slot churn and runner restarts.

- **`SlotLRU`** (`proxy/router.py`) — `OrderedDict` session_id → slot_id with capacity equal to the model's `--parallel`. First N distinct sessions claim slots `0..N-1`; further sessions LRU-evict the oldest. `touch(session_id) -> (slot_id, evicted_pair)` is the only mutator; `peek()` does not affect recency. One `SlotLRU` per upstream `server_id` (held in `_slot_lrus`).
- **Upstream body injection** — Before forwarding a chat completion the proxy sets `id_slot`, `cache_prompt=true`, and `n_cache_reuse=256` on the JSON body so llama.cpp lands the request in our pinned slot and reuses the prefix cache.
- **Save-on-evict / restore-before-use** — When `touch()` returns an `evicted_pair`, the evicted session's slot is saved to disk via the llama.cpp `/slots/{id}?action=save` endpoint. Before the new session's first request the proxy attempts a `/slots/{id}?action=restore` from disk if a file exists for that session.
- **Slot file naming** — `slot_<session_id>_<model_id>.bin` under `SLOT_SAVE_DIR`. `model_id` is resolved via `_stable_model_key()` from `server_cache` so the filename survives runner pod restarts (the old `server_id`-keyed naming was broken because `server_id` is an ephemeral UUID per subprocess). Different models still get different files to prevent cross-model restore (incompatible tensor shapes). Lookup is best-effort: if `server_cache` doesn't know the `server_id`, fall back to the legacy path so nothing crashes.
- **Pooled httpx clients** — One `httpx.AsyncClient` per upstream server (in `_upstream_clients`) with limits keyed off `--parallel`. Replaces the per-request client pattern; preserves TCP keep-alive across forwards. `aclose_all_clients()` closes the pool and is wired into `app.py` lifespan shutdown between background-task cancellation and `server_cache.stop_all()` so connections drain before subprocesses are killed.
- **Slot cleanup** — Background task in `app.py` periodically reaps old `slot_*.bin` files. `SLOT_INACTIVE_MAX_AGE_MIN` (default 12 min) checks the proxy's session-activity tracker; `SLOT_CLEANUP_MAX_AGE_MIN` (default 12 min) is an absolute floor; `SLOT_CLEANUP_MAX_SIZE_MB` (default 5000 MB) enforces a directory size cap. Old `slot_<session>_<server_id>.bin` files from before the model_id rename are reaped via the size/age caps.
- **Invariants** — Slot errors must never break the main request flow (all wrapped in try/except). Use `_resolve_slot_id` only for backward-compat tests — production code goes through `SlotLRU`.

### Prompt Fingerprint Diagnostic

`proxy/router.py::_log_prompt_divergence()` hashes the request body at offsets `[1K, 2K, 4K, 8K, 12K, 16K, 20K, 24K, 32K, 48K, 64K, 96K, 128K]` per session before slot/cache injection and logs one of:

- `FIRST` — first turn for a session
- `STABLE` — all hashes match the previous turn (prompt prefix is byte-stable, good for cache reuse)
- `DIVERGED at byte N` — first offset where hashes diverged, indicating where the upstream caller mutated the prompt

Used to diagnose why llama.cpp's prefix cache wasn't matching as far as expected. Prometheus counterparts: `prompt_fingerprint_total{kind=first|stable|diverged}` and `prompt_first_divergence_byte` (histogram).

### Process-Death Watchdog

Each `BaseServerManager` (`server_manager/base.py`) spawns a watchdog thread (`_watchdog_run`) that `proc.wait()`s on the llama.cpp subprocess. On unexpected exit (returncode not in `{0, -15, 143}` and `_intentional_stop` is false), the watchdog:

1. Snapshots the last 50 lines from the stderr ring buffer (200-line buffer per manager, drained at INFO so live CUDA errors surface immediately rather than appearing only as exit code -9).
2. Logs WARNING with the stderr tail and exit code.
3. Calls `ServerCache.purge_dead_server(server_id)` — idempotent — which drops the cache entry immediately (no waiting for the 30-60 min TTL eviction) and bumps `llama_server_evictions_total{reason="process_died"}`.

`_intentional_stop` is set to True by `stop()`, the retry-with-reduced-context path, and `ServerCache` eviction before terminating the process so the watchdog distinguishes "we asked it to exit" from "it died."  `attach_lifecycle()` wires the cache reference into the manager so the watchdog can purge without an import cycle.

### GET /v1/status

`routers/status.py` exposes a cheap liveness/identity endpoint:

```json
{ "startup_epoch": <unix_ms>, "now": <unix_ms> }
```

`startup_epoch` is captured at module import. The api side polls this to detect runner restarts: when the epoch changes, the api invalidates all cached `server_id` handles and reacquires them. No auth, no I/O — safe to call frequently.

### Prometheus Metrics (Round 1)

Exposed at `/metrics` (`middleware/prometheus_metrics.py`). Cardinality is intentionally bounded — no session_id labels.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `slot_resolutions_total` | counter | `slot_id`, `evicted` | One per `SlotLRU.touch()`; `evicted=true` when the touch displaced an existing session |
| `slot_lru_size{server_id}` | gauge | `server_id` | Current SlotLRU population per upstream |
| `slot_save_total` / `slot_restore_total` | counter | `slot_id`, `outcome` (`success`/`failure`) | Save-on-evict / restore-before-use outcomes |
| `slot_save_duration_seconds` / `slot_restore_duration_seconds` | histogram | — | Wall-clock of the upstream `/slots/{id}?action=...` call |
| `prompt_fingerprint_total` | counter | `kind` (`first`/`stable`/`diverged`) | One per chat completion, from the fingerprint diagnostic |
| `prompt_first_divergence_byte` | histogram | — | Offset of first mismatched fingerprint hash; lower buckets mean worse cache reuse |
| `prompt_body_bytes` | histogram | — | Request body size distribution |
| `llama_server_evictions_total{reason="process_died"}` | counter | `reason` | Now fires from the watchdog on unexpected subprocess exit |

### Launch Flag: --cache-reuse

`server_manager/llamacpp_argument_builder.py` passes `--cache-reuse N` to `llama-server` for KV-shift prefix reuse. The value comes from the per-model `parameters.cache_reuse` (in `.models.yaml`, modelled in `models/model_parameters.py::ModelParameters.cache_reuse`); default is 256 when unset. Required for per-session prompt-cache reuse to take effect alongside `n_cache_reuse` in the request body.

### Model Config Notes (.models.yaml)

- **`Qwen3_6_27B.split_mode: layer`** — `row` triggers `GGML_ASSERT(!(split && ne02 < ne12))` in `ggml-cuda.cu` when two slots pack into the same ubatch, taking down the whole `llama-server` process. `layer` partitions by layer, tolerates concurrent slots, slight cross-GPU bandwidth trade-off.
- **`Qwen3_6_27B.micro_batch_size: 512`** — With `ubatch_size=2048` a single slot's 2048-token chunk monopolised each micro-batch, blocking the second slot's prefill for minutes. 512 lets multiple slots share each ubatch. `batch_size` stays at 2048 so total throughput per cycle is preserved.

### Session ID Middleware

- `SessionIdMiddleware` in `app.py` extracts `X-Session-ID` header into `_session_id_ctx` context variable
- Session ID flows through structlog entries automatically via `_add_session_id_to_logs` processor
- Accessible in proxy router via `_session_id_ctx.get()`

### Logging Infrastructure

- Loki is exposed through the gateway load balancer at `http://192.168.0.71:3100`
- Query API: `http://192.168.0.71:3100/loki/api/v1/query_range`
- Use `{namespace="llmmllab"}` selector for runner logs
- Logs are stored on a PVC so they survive pod restarts
