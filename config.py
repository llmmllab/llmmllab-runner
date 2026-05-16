import os
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "console")
RUNNER_NAME = os.environ.get("RUNNER_NAME", "llmmllab-runner")
LLAMA_SERVER_EXECUTABLE = os.environ.get(
    "LLAMA_SERVER_EXECUTABLE", "/llama.cpp/build/bin/llama-server"
)
MODELS_FILE_PATH = os.environ.get("MODELS_FILE_PATH", "")

# Soft timeout (minutes): once a server has been idle (use_count == 0) for this
# long, it becomes *eligible* for eviction if a new request needs VRAM space.
CACHE_TIMEOUT_MIN = int(os.environ.get("CACHE_TIMEOUT_MIN", "30"))

# Hard timeout (minutes): once a server has been idle for this long, it *must*
# be evicted regardless of VRAM pressure.
EVICTION_TIMEOUT_MIN = int(os.environ.get("EVICTION_TIMEOUT_MIN", "60"))

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
DCGM_EXPORTER_URL = os.environ.get(
    "DCGM_EXPORTER_URL", "http://localhost:9400/metrics"
)

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

# When SLOT_SAVE_DIR is set, also pass --no-mmap to llama-server.
# Memory-mapped model loading can interfere with slot persistence
# because the OS may evict mmap pages between save and restore.
# Set to "false" to disable even when slot persistence is enabled.
SLOT_NO_MMAP = os.environ.get("SLOT_NO_MMAP", "true").lower() in (
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
SLOT_CLEANUP_MAX_AGE_MIN = int(os.environ.get("SLOT_CLEANUP_MAX_AGE_MIN", "1440"))

# Slot file cleanup: maximum total size of /slots directory in MB.
# When exceeded, oldest files are deleted until under the limit.
# Set to 0 to disable size-based cleanup.
SLOT_CLEANUP_MAX_SIZE_MB = int(os.environ.get("SLOT_CLEANUP_MAX_SIZE_MB", "5000"))

# How often (seconds) the slot cleanup task runs.
SLOT_CLEANUP_INTERVAL_SEC = int(os.environ.get("SLOT_CLEANUP_INTERVAL_SEC", "300"))
