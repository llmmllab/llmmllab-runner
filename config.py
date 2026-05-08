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
