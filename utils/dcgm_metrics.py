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

from config import DCGM_EXPORTER_URL, DCGM_METRICS_ENABLED, NODE_NAME
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="DCGMMetrics")

logger = llmmllogger.bind(component="DCGMMetrics")


# Mapping of NVIDIA dcgm-exporter Prometheus metric names to our
# internal Prometheus gauge keys (see middleware/prometheus_metrics.py
# for the gauge definitions).
#
# Real DCGM exporter metric names use the DCGM_FI_DEV_* prefix.  The
# previous mapping used nv_* names which never existed in dcgm-exporter
# output — so the scrape "succeeded" in HTTP terms but mapped zero
# samples to gauges, silently emitting nothing.
_DCGM_METRIC_MAP: Dict[str, str] = {
    # GPU utilization
    "DCGM_FI_DEV_GPU_UTIL": "gpu_compute_utilization",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "gpu_memory_bandwidth_utilization",
    # Clocks
    "DCGM_FI_DEV_SM_CLOCK": "gpu_sm_clock_mhz",
    "DCGM_FI_DEV_MEM_CLOCK": "gpu_mem_clock_mhz",
    # Power
    "DCGM_FI_DEV_POWER_USAGE": "gpu_power_watts",
    # Encoder / Decoder utilization
    "DCGM_FI_DEV_DEC_UTIL": "gpu_decoder_utilization",
    "DCGM_FI_DEV_ENC_UTIL": "gpu_encoder_utilization",
    # ECC: dcgm-exporter exposes counters by error type (volatile/aggregate,
    # SBE/DBE).  These names match common configurations; if your DCGM
    # config doesn't enable them, they're simply absent and harmless.
    "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "gpu_ecc_sbe_volatile_total",
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": "gpu_ecc_dbe_volatile_total",
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
    except httpx.ConnectError as e:
        logger.warning(
            f"DCGM exporter connection failed: {e}. "
            f"Ensure DCGM exporter sidecar is running at {DCGM_EXPORTER_URL}"
        )
    except httpx.HTTPError as e:
        logger.warning(f"DCGM exporter HTTP error: {e}")
    except Exception as e:
        logger.warning(f"DCGM scrape failed: {e}")
        return {}

    samples = _parse_prometheus_text(raw)

    # Build per-GPU metric dicts, filtered to the local node only.
    #
    # The dcgm-exporter Service load-balances across all GPU nodes.  Without
    # a Hostname filter, alternating scrapes return metrics for whichever
    # node the Service picked, causing gauge values to flicker between
    # nodes' GPUs.  Drop samples whose Hostname label doesn't match this
    # pod's NODE_NAME (set via the downward API in deployment.yaml).
    #
    # If NODE_NAME isn't set (local dev, manual run), accept all samples.
    gpu_metrics: Dict[str, Dict[str, float]] = {}
    target_host = NODE_NAME or ""
    foreign_dropped = 0

    for dcgm_name, internal_key in _DCGM_METRIC_MAP.items():
        for sample in samples.get(dcgm_name, []):
            if target_host and sample.get("Hostname", "") != target_host:
                foreign_dropped += 1
                continue
            gpu_idx = sample.get("gpu", sample.get("GPU", "0"))
            value = sample["__value__"]
            gpu_metrics.setdefault(str(gpu_idx), {})[internal_key] = value

    if gpu_metrics:
        logger.debug(
            f"DCGM scrape: {len(gpu_metrics)} GPUs, "
            f"{sum(len(v) for v in gpu_metrics.values())} metrics"
            + (f" (dropped {foreign_dropped} from foreign nodes)" if foreign_dropped else "")
        )
    elif foreign_dropped:
        # All samples came from a different node.  Common when the Service
        # routes the scrape to a non-local exporter.  Log at debug, not
        # warning — next scrape may pick the local one.
        logger.debug(
            f"DCGM scrape returned only foreign-node metrics "
            f"({foreign_dropped} dropped, target={target_host!r}). "
            "Service may have load-balanced to another node."
        )
    return gpu_metrics
