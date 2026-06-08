import os
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "console")
RUNNER_NAME = os.environ.get("RUNNER_NAME", "llmmllab-runner")
LLAMA_SERVER_EXECUTABLE = os.environ.get(
    "LLAMA_SERVER_EXECUTABLE", "/llama.cpp/build/bin/llama-server"
)
SD_SERVER_EXECUTABLE = os.environ.get(
    "SD_SERVER_EXECUTABLE", "/stable-diffusion.cpp/build/bin/sd-server"
)
# Image-output directory used by both the SD server manager (when run locally)
# and the API layer for serving generated images.  The runner only uses this
# for housekeeping/cleanup; the actual images are returned base64-encoded.
SD_OUTPUT_DIR = os.environ.get("SD_OUTPUT_DIR", "/data/sd-out")
MODELS_FILE_PATH = os.environ.get("MODELS_FILE_PATH", "")

# Soft timeout (minutes): once a server has been idle (use_count == 0) for this
# long, it becomes *eligible* for eviction if a new request needs VRAM space.
# Also used by the api as the cutoff for "this server might still belong to a
# paused session, don't commandeer it" vs "safe to reuse for a new session."
# Was 30 → dropped to 5 → bumped to 10. 5 was too tight: a user pausing for
# 6-7 minutes to type a reply would lose their slot to a cron commandeer.
# 10 keeps interactive multi-turn sessions sticky long enough to survive a
# coffee/Slack-side-conversation pause without sacrificing peer fan-out
# (the api-side selector now prefers an empty peer over any *loaded* peer,
# warm-idle included, so an idle peer still picks up new traffic regardless
# of this window).
CACHE_TIMEOUT_MIN = int(os.environ.get("CACHE_TIMEOUT_MIN", "10"))

# Hard timeout (minutes): once a server has been idle for this long, it *must*
# be evicted regardless of VRAM pressure.  Was 60; dropped to 30 to match the
# tightened CACHE_TIMEOUT_MIN (was a 2× multiple, kept as such here).
EVICTION_TIMEOUT_MIN = int(os.environ.get("EVICTION_TIMEOUT_MIN", "30"))

RUNNER_PORT = int(os.environ.get("RUNNER_PORT", "8000"))
RUNNER_HOST = os.environ.get("RUNNER_HOST", "0.0.0.0")
SERVER_PORT_RANGE_START = int(os.environ.get("SERVER_PORT_RANGE_START", "8001"))
SERVER_PORT_RANGE_END = int(os.environ.get("SERVER_PORT_RANGE_END", "8900"))

# Proxy timeout (seconds) for upstream llama.cpp requests.
# Must exceed the longest expected inference time.  Streaming requests
# hold the connection open for the entire generation.
PROXY_TIMEOUT = float(os.environ.get("PROXY_TIMEOUT", "600"))

# GPU power cap: percentage of default TDP (0 to disable, 100 = no cap)
GPU_POWER_CAP_PCT = float(os.environ.get("GPU_POWER_CAP_PCT", "85"))

# Max retries when a llama.cpp server start fails due to OOM (SIGKILL / exit -9).
# Set to 0 to disable OOM retries.
SERVER_START_OOM_RETRIES = int(os.environ.get("SERVER_START_OOM_RETRIES", "2"))

# DCGM Exporter integration
DCGM_METRICS_ENABLED = os.environ.get("DCGM_METRICS_ENABLED", "true").lower() in (
    "true",
    "1",
    "yes",
)
# NVIDIA gpu-operator deploys dcgm-exporter as a DaemonSet under the
# gpu-operator namespace with a ClusterIP service.  The service load-balances
# across all GPU nodes' exporters, so callers must filter by NODE_NAME (see
# utils/dcgm_metrics.py) to scrape only their local node's GPUs.  The old
# default of localhost:9400 assumed a sidecar pattern that doesn't exist.
DCGM_EXPORTER_URL = os.environ.get(
    "DCGM_EXPORTER_URL",
    "http://nvidia-dcgm-exporter.gpu-operator.svc.cluster.local:9400/metrics",
)

# Set via the downward API in k8s/deployment.yaml (spec.nodeName).
# Used by utils/dcgm_metrics.py to filter DCGM metrics to the local
# node only, since the dcgm-exporter Service load-balances across
# all GPU nodes.
NODE_NAME = os.environ.get("NODE_NAME", "")

# Llama.cpp server metrics scraping interval (seconds)
LLAMA_METRICS_INTERVAL_SEC = int(
    os.environ.get("LLAMA_METRICS_INTERVAL_SEC", "15")
)

# DCGM metrics scrape interval (seconds)
DCGM_METRICS_INTERVAL_SEC = int(
    os.environ.get("DCGM_METRICS_INTERVAL_SEC", "15")
)

# Persistent KV cache slot directory for session persistence.
# When set, the runner passes --slot-save-path to llama-server so that
# conversation state (KV cache) can be saved/restored via the llama.cpp
# REST API (/slots/{id}/save, /slots/{id}/restore).  Each slot file is
# named by slot index and lives under this directory.
# Example: SLOT_SAVE_DIR=/data/slots  →  --slot-save-path /data/slots
SLOT_SAVE_DIR = os.environ.get("SLOT_SAVE_DIR", "")

# --no-mmap was historically forced on whenever SLOT_SAVE_DIR was set,
# based on the upstream slot-persistence tutorial
# (https://github.com/ggml-org/llama.cpp/discussions/20572) which lists
# it under "Avoids memory-mapped file issues on some platforms" —
# a platform-specific workaround, NOT a slot-persistence requirement.
#
# The architectural requirements for safe slot KV save/restore are:
#   --slot-save-path <dir>     enables the /slots/<id>?action=save|restore endpoints
#   --swa-full                 required for SWA models (Qwen 3.x) so the sliding
#                              window doesn't corrupt persisted KV state
#   cache_prompt: true         in the request body, so the server matches existing
#                              prefix cache after restore
# All three are configured elsewhere; --no-mmap is unrelated.
#
# Default is "false" because --no-mmap forces llama.cpp to load the
# entire model file (~35 GB for Qwen3.6-27B Q6) into a malloc'd host
# buffer before GPU transfer, spiking host RSS to model-size during
# model load / switch and OOMKilling memory-limited pods. With mmap
# (the OS default), only actively-touched pages stay resident — host
# RSS during steady state is well under 2 GiB regardless of model size.
#
# Set SLOT_NO_MMAP=true to opt back in only if you hit a specific
# platform issue with slot save/restore when mmap is enabled.
SLOT_NO_MMAP = os.environ.get("SLOT_NO_MMAP", "false").lower() in (
    "true",
    "1",
    "yes",
)

# When SLOT_SAVE_DIR is set, also pass --swa-full to llama-server.
# This is required for SWA (Sliding Window Attention) models such as
# Qwen 3.5 to correctly persist and restore their KV cache.
# Set to "false" to disable even when slot persistence is enabled.
SLOT_SWA_FULL = os.environ.get("SLOT_SWA_FULL", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Slot file cleanup: delete slot files older than this many minutes.
# Only active when SLOT_SAVE_DIR is set. Set to 0 to disable cleanup.
SLOT_CLEANUP_MAX_AGE_MIN = int(os.environ.get("SLOT_CLEANUP_MAX_AGE_MIN", "12"))

# Slot file cleanup: maximum total size of /slots directory in MB.
# When exceeded, oldest files are deleted until under the limit.
# Set to 0 to disable size-based cleanup.
SLOT_CLEANUP_MAX_SIZE_MB = int(os.environ.get("SLOT_CLEANUP_MAX_SIZE_MB", "5000"))

# How often (seconds) the slot cleanup task runs.
SLOT_CLEANUP_INTERVAL_SEC = int(os.environ.get("SLOT_CLEANUP_INTERVAL_SEC", "300"))

# Slot file cleanup: only delete slot files for sessions that haven't had a
# turn in this many minutes.  Unlike SLOT_CLEANUP_MAX_AGE_MIN (which uses
# file mtime), this looks at the proxy's session-activity tracker so that
# slot files for sessions that are still "alive" but idle between turns
# aren't prematurely deleted.  Set to 0 to disable (fall back to mtime).
SLOT_INACTIVE_MAX_AGE_MIN = int(os.environ.get("SLOT_INACTIVE_MAX_AGE_MIN", "12"))
