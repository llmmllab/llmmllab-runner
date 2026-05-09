"""Llama.cpp server metrics scraper.

Scrapes /metrics from each active llama.cpp server instance and
parses the Prometheus-format output into structured per-server
metrics for exposition by the runner's Prometheus middleware.

Llama.cpp must be compiled with metrics support (most recent builds
include this by default).  If a server's /metrics endpoint is not
available, that server is silently skipped.
"""

from typing import Any, Dict, List, Optional

import httpx

from cache import ServerCache
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="LlamaMetrics")


# Mapping of llama.cpp metric names to our internal gauge keys.
_LLAMA_METRIC_MAP: Dict[str, str] = {
    "tps": "tokens_per_second",
    "prompt_t_ps": "prompt_tokens_per_second",
    "t_token_ms": "ms_per_token",
    "t_p_token_ms": "ms_per_prompt_token",
    "n_eval": "tokens_evaluated_total",
    "n_predict": "tokens_predicted_total",
    "n_vocab": "vocab_size",
    "n_ctx": "context_size",
}


def _parse_prometheus_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse Prometheus text-format metrics into a dict of samples."""
    samples: Dict[str, List[Dict[str, Any]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        brace_idx = line.find("{")
        if brace_idx >= 0:
            name = line[:brace_idx]
            close_brace = line.index("}", brace_idx)
            label_str = line[close_brace + 1 :].strip()
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            label_str = parts[1]

        try:
            value = float(label_str)
        except ValueError:
            continue

        samples.setdefault(name, []).append({"__value__": value})

    return samples


async def scrape_server_metrics(
    server_id: str, port: int, model_id: str
) -> Dict[str, float]:
    """Scrape /metrics from a single llama.cpp server.

    Returns a dict of metric_name → value.  Returns empty dict on error.
    """
    url = f"http://127.0.0.1:{port}/metrics"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw = resp.text
    except Exception as e:
        logger.debug(f"Server {server_id} metrics scrape failed: {e}")
        return {}

    samples = _parse_prometheus_text(raw)

    metrics: Dict[str, float] = {}
    for llama_name, internal_key in _LLAMA_METRIC_MAP.items():
        for sample in samples.get(llama_name, []):
            metrics[internal_key] = sample["__value__"]

    # Also check for slot-level metrics (prefixed with slot_)
    for sample_name, sample_list in samples.items():
        if sample_name.startswith("slot_"):
            for sample in sample_list:
                metrics[f"slot_{sample_name}"] = sample["__value__"]

    if metrics:
        logger.debug(
            f"Server {server_id} ({model_id}): {len(metrics)} metrics scraped"
        )
    return metrics


async def scrape_all_server_metrics(cache: ServerCache) -> Dict[str, Dict[str, float]]:
    """Scrape metrics from all active servers in the cache.

    Returns a dict keyed by server_id, each value is a dict of
    metric_name → value.
    """
    stats = cache.stats()
    results: Dict[str, Dict[str, float]] = {}

    for server_info in stats.get("servers", []):
        server_id = server_info["server_id"]
        port = server_info["port"]
        model_id = server_info["model_id"]

        if server_info.get("starting"):
            continue  # Skip servers still starting

        metrics = await scrape_server_metrics(server_id, port, model_id)
        if metrics:
            results[server_id] = metrics

    return results
