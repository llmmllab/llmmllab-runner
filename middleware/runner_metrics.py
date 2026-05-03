"""Business-level metrics helpers for the runner service."""

from cache import ServerCache
from config import RUNNER_NAME
from middleware.prometheus_metrics import (
    llama_servers_active,
    llama_server_starts_total,
    llama_server_evictions_total,
    gpu_temperature_celsius,
    gpu_memory_used_bytes,
    gpu_power_watts,
)
from utils.hardware_manager import hardware_manager


def update_server_metrics(cache: ServerCache):
    """Update server count gauge. Call periodically."""
    stats = cache.stats()
    llama_servers_active.labels(runner_name=RUNNER_NAME).set(
        stats["active_servers"]
    )


def record_server_start(model_id: str):
    """Call when a new llama.cpp server is created."""
    llama_server_starts_total.labels(model_id=model_id).inc()


def record_server_eviction(reason: str):
    """Call when a server is evicted. reason: 'idle', 'vram_pressure', 'manual'."""
    llama_server_evictions_total.labels(reason=reason).inc()


def update_gpu_metrics():
    """Update GPU temperature, memory, and power gauges. Call periodically."""
    stats = hardware_manager.gpu_stats()
    for gpu_id, info in stats.items():
        idx = gpu_id
        gpu_memory_used_bytes.labels(gpu_index=idx).set(
            info.get("used_mb", 0) * 1024 * 1024
        )

    # Temperature comes from the thermal check
    temps = hardware_manager.check_gpu_thermals()
    for idx, temp in temps.items():
        gpu_temperature_celsius.labels(gpu_index=str(idx)).set(temp)

    # Power — query via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                gpu_power_watts.labels(gpu_index=str(i)).set(float(line.strip()))
    except Exception:
        pass
