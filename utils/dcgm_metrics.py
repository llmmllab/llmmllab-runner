"""NVIDIA DCGM Exporter metrics scraper.

Fetches Prometheus-format metrics from the DCGM exporter sidecar
(http://localhost:9400/metrics by default) and exposes them as
structured dicts for consumption by the Prometheus middleware.

DCGM exporter must be running as a sidecar container or background
process.  If it's unavailable, all scrapes return empty dicts and
the runner continues operating normally.
"""

from typing import Any, Dict, List, Optional

import httpx

from config import DCGM_EXPORTER_URL, DCGM_METRICS_ENABLED
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="DCGMMetrics")


# Mapping of DCGM Prometheus metric names to our internal keys.
# DCGM labels include `GPU` (UUID) and `gpu` (index).
_DCGM_METRIC_MAP: Dict[str, str] = {
    # GPU utilization
    "nv_gpu_utilization": "gpu_compute_utilization",
    "nv_mem_copy_utilization": "gpu_memory_bandwidth_utilization",
    # Clocks
    "nv_sm_clock": "gpu_sm_clock_mhz",
    "nv_mem_clock": "gpu_mem_clock_mhz",
    # Fan
    "nv_fan_speed": "gpu_fan_speed_percent",
    # Power (DCGM version — may overlap with nvidia-smi path)
    "nv_power_usage": "gpu_power_watts",
    # ECC
    "nv_ecc_sbe_volatile": "gpu_ecc_sbe_volatile_total",
    "nv_ecc_dbe_volatile": "gpu_ecc_dbe_volatile_total",
    # PCIe throughput
    "nv_pcie_tx_bytes": "gpu_pcie_tx_bytes_total",
    "nv_pcie_rx_bytes": "gpu_pcie_rx_bytes_total",
    # NVLink (if applicable)
    "nv_nvlink_flit_tx": "gpu_nvlink_flit_tx_total",
    "nv_nvlink_flit_rx": "gpu_nvlink_flit_rx_total",
    # Encoder / Decoder utilization
    "nv_vbios_version": "nv_vbios_version",  # informational
    "nv_dec_utilization": "gpu_decoder_utilization",
    "nv_enc_utilization": "gpu_encoder_utilization",
}


def _parse_prometheus_text(text: str) -> Dict[str, Dict[str, str]]:
    """Parse Prometheus text-format metrics into a dict of samples.

    Returns a dict keyed by metric name, each value is a list of
    (labels_dict, value) tuples.
    """
    samples: Dict[str, List[Dict[str, Any]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Format: metric_name{label="value",...} value
        # or:    metric_name value
        brace_idx = line.find("{")
        if brace_idx >= 0:
            name = line[:brace_idx]
            close_brace = line.index("}", brace_idx)
            label_str = line[brace_idx + 1 : close_brace]
            value_str = line[close_brace + 1 :].strip()
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            value_str = parts[1]
            label_str = ""

        try:
            value = float(value_str)
        except ValueError:
            continue

        labels: Dict[str, str] = {}
        if label_str:
            for part in label_str.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')

        labels["__value__"] = value
        samples.setdefault(name, []).append(labels)

    return samples


async def scrape_dcgm_metrics() -> Dict[str, Dict[str, float]]:
    """Scrape DCGM exporter and return per-GPU metrics.

    Returns a dict keyed by GPU index (string), each value is a dict
    of metric_name → value.  Returns empty dict on any error.
    """
    if not DCGM_METRICS_ENABLED:
        return {}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(DCGM_EXPORTER_URL)
            resp.raise_for_status()
            raw = resp.text
    except Exception as e:
        logger.debug(f"DCGM scrape failed: {e}")
        return {}

    samples = _parse_prometheus_text(raw)

    # Build per-GPU metric dicts
    gpu_metrics: Dict[str, Dict[str, float]] = {}

    for dcgm_name, internal_key in _DCGM_METRIC_MAP.items():
        for sample in samples.get(dcgm_name, []):
            gpu_idx = sample.get("gpu", sample.get("GPU", "0"))
            value = sample["__value__"]
            gpu_metrics.setdefault(str(gpu_idx), {})[internal_key] = value

    if gpu_metrics:
        logger.debug(f"DCGM scrape: {len(gpu_metrics)} GPUs, "
                      f"{sum(len(v) for v in gpu_metrics.values())} metrics")
    return gpu_metrics
