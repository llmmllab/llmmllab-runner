"""Business-level metrics helpers for the runner service."""

from typing import Dict

from cache import ServerCache
from config import RUNNER_NAME
from middleware.prometheus_metrics import (
    llama_servers_active,
    llama_server_starts_total,
    llama_server_evictions_total,
    gpu_temperature_celsius,
    gpu_memory_used_bytes,
    gpu_power_watts,
    gpu_compute_utilization,
    gpu_memory_bandwidth_utilization,
    gpu_sm_clock_mhz,
    gpu_mem_clock_mhz,
    gpu_fan_speed_percent,
    gpu_ecc_errors_total,
    gpu_encoder_utilization,
    gpu_decoder_utilization,
    llama_server_tokens_per_second,
    llama_server_prompt_tokens_per_second,
    llama_server_ms_per_token,
    llama_server_tokens_evaluated_total,
    llama_server_tokens_predicted_total,
)
from utils.hardware_manager import hardware_manager
from utils.dcgm_metrics import scrape_dcgm_metrics
from utils.llama_metrics import scrape_all_server_metrics


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


def _apply_dcgm_metrics(dcgm_data: Dict[str, Dict[str, float]]) -> None:
    """Apply DCGM-scraped metrics to Prometheus gauges."""
    for gpu_idx, metrics in dcgm_data.items():
        if "gpu_compute_utilization" in metrics:
            gpu_compute_utilization.labels(gpu_index=gpu_idx).set(
                metrics["gpu_compute_utilization"]
            )
        if "gpu_memory_bandwidth_utilization" in metrics:
            gpu_memory_bandwidth_utilization.labels(gpu_index=gpu_idx).set(
                metrics["gpu_memory_bandwidth_utilization"]
            )
        if "gpu_sm_clock_mhz" in metrics:
            gpu_sm_clock_mhz.labels(gpu_index=gpu_idx).set(
                metrics["gpu_sm_clock_mhz"]
            )
        if "gpu_mem_clock_mhz" in metrics:
            gpu_mem_clock_mhz.labels(gpu_index=gpu_idx).set(
                metrics["gpu_mem_clock_mhz"]
            )
        if "gpu_fan_speed_percent" in metrics:
            gpu_fan_speed_percent.labels(gpu_index=gpu_idx).set(
                metrics["gpu_fan_speed_percent"]
            )
        if "gpu_power_watts" in metrics:
            gpu_power_watts.labels(gpu_index=gpu_idx).set(
                metrics["gpu_power_watts"]
            )
        if "gpu_ecc_sbe_volatile_total" in metrics:
            gpu_ecc_errors_total.labels(gpu_index=gpu_idx, type="sbe_volatile").set(
                metrics["gpu_ecc_sbe_volatile_total"]
            )
        if "gpu_ecc_dbe_volatile_total" in metrics:
            gpu_ecc_errors_total.labels(gpu_index=gpu_idx, type="dbe_volatile").set(
                metrics["gpu_ecc_dbe_volatile_total"]
            )
        if "gpu_encoder_utilization" in metrics:
            gpu_encoder_utilization.labels(gpu_index=gpu_idx).set(
                metrics["gpu_encoder_utilization"]
            )
        if "gpu_decoder_utilization" in metrics:
            gpu_decoder_utilization.labels(gpu_index=gpu_idx).set(
                metrics["gpu_decoder_utilization"]
            )


def update_gpu_metrics() -> None:
    """Update GPU temperature, memory, and power gauges. Call periodically.

    Uses nvsmi for VRAM, nvidia-smi for temperature, and falls back to
    nvidia-smi for power if DCGM is not available.
    """
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

    # Power — query via nvidia-smi (fallback if DCGM not available)
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


async def update_dcgm_metrics() -> None:
    """Scrape DCGM exporter and update additional GPU gauges.

    This is an async function intended to be called from the background
    metrics task in app.py.  It supplements update_gpu_metrics() with
    richer metrics from the DCGM exporter.
    """
    dcgm_data = await scrape_dcgm_metrics()
    if dcgm_data:
        _apply_dcgm_metrics(dcgm_data)


async def update_llama_server_metrics(cache: ServerCache) -> None:
    """Scrape /metrics from all active llama.cpp servers and update gauges.

    This is an async function intended to be called from the background
    metrics task in app.py.
    """
    all_metrics = await scrape_all_server_metrics(cache)
    stats = cache.stats()
    server_map = {s["server_id"]: s for s in stats.get("servers", [])}

    for server_id, metrics in all_metrics.items():
        server_info = server_map.get(server_id, {})
        model_id = server_info.get("model_id", "unknown")

        if "tokens_per_second" in metrics:
            llama_server_tokens_per_second.labels(
                server_id=server_id, model_id=model_id
            ).set(metrics["tokens_per_second"])
        if "prompt_tokens_per_second" in metrics:
            llama_server_prompt_tokens_per_second.labels(
                server_id=server_id, model_id=model_id
            ).set(metrics["prompt_tokens_per_second"])
        if "ms_per_token" in metrics:
            llama_server_ms_per_token.labels(
                server_id=server_id, model_id=model_id
            ).set(metrics["ms_per_token"])
        if "tokens_evaluated_total" in metrics:
            llama_server_tokens_evaluated_total.labels(
                server_id=server_id, model_id=model_id
            ).set(metrics["tokens_evaluated_total"])
        if "tokens_predicted_total" in metrics:
            llama_server_tokens_predicted_total.labels(
                server_id=server_id, model_id=model_id
            ).set(metrics["tokens_predicted_total"])
