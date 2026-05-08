"""Prometheus metrics middleware and GPU/server metrics for the runner service."""

import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_registry = CollectorRegistry()

# HTTP metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=_registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=_registry,
)

active_http_connections = Gauge(
    "http_active_connections",
    "Number of active HTTP connections",
    registry=_registry,
)

# Runner-specific metrics
llama_servers_active = Gauge(
    "llama_servers_active",
    "Number of active llama.cpp servers",
    ["runner_name"],
    registry=_registry,
)

llama_server_starts_total = Counter(
    "llama_server_starts_total",
    "Total llama.cpp server starts",
    ["model_id"],
    registry=_registry,
)

llama_server_evictions_total = Counter(
    "llama_server_evictions_total",
    "Total llama.cpp server evictions",
    ["reason"],
    registry=_registry,
)

llama_server_proxy_errors_total = Counter(
    "llama_server_proxy_errors_total",
    "Total proxy errors to llama.cpp servers",
    ["status"],
    registry=_registry,
)

gpu_temperature_celsius = Gauge(
    "gpu_temperature_celsius",
    "GPU temperature in celsius",
    ["gpu_index"],
    registry=_registry,
)

gpu_memory_used_bytes = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_index"],
    registry=_registry,
)

gpu_power_watts = Gauge(
    "gpu_power_watts",
    "GPU power usage in watts",
    ["gpu_index"],
    registry=_registry,
)

# --- Additional GPU metrics from DCGM Exporter ---

gpu_compute_utilization = Gauge(
    "gpu_compute_utilization_percent",
    "GPU compute utilization percentage",
    ["gpu_index"],
    registry=_registry,
)

gpu_memory_bandwidth_utilization = Gauge(
    "gpu_memory_bandwidth_utilization_percent",
    "GPU memory bandwidth utilization percentage",
    ["gpu_index"],
    registry=_registry,
)

gpu_sm_clock_mhz = Gauge(
    "gpu_sm_clock_mhz",
    "GPU SM clock speed in MHz",
    ["gpu_index"],
    registry=_registry,
)

gpu_mem_clock_mhz = Gauge(
    "gpu_mem_clock_mhz",
    "GPU memory clock speed in MHz",
    ["gpu_index"],
    registry=_registry,
)

gpu_fan_speed_percent = Gauge(
    "gpu_fan_speed_percent",
    "GPU fan speed percentage",
    ["gpu_index"],
    registry=_registry,
)

gpu_ecc_errors_total = Gauge(
    "gpu_ecc_errors_total",
    "GPU ECC error count",
    ["gpu_index", "type"],  # type: "sbe_volatile", "dbe_volatile"
    registry=_registry,
)

gpu_encoder_utilization = Gauge(
    "gpu_encoder_utilization_percent",
    "GPU video encoder utilization percentage",
    ["gpu_index"],
    registry=_registry,
)

gpu_decoder_utilization = Gauge(
    "gpu_decoder_utilization_percent",
    "GPU video decoder utilization percentage",
    ["gpu_index"],
    registry=_registry,
)

# --- Llama.cpp server metrics ---

llama_server_tokens_per_second = Gauge(
    "llama_server_tokens_per_second",
    "Token generation rate (tokens/s)",
    ["server_id", "model_id"],
    registry=_registry,
)

llama_server_prompt_tokens_per_second = Gauge(
    "llama_server_prompt_tokens_per_second",
    "Prompt evaluation rate (tokens/s)",
    ["server_id", "model_id"],
    registry=_registry,
)

llama_server_ms_per_token = Gauge(
    "llama_server_ms_per_token",
    "Milliseconds per generated token",
    ["server_id", "model_id"],
    registry=_registry,
)

llama_server_tokens_evaluated_total = Gauge(
    "llama_server_tokens_evaluated_total",
    "Total tokens evaluated by the server",
    ["server_id", "model_id"],
    registry=_registry,
)

llama_server_tokens_predicted_total = Gauge(
    "llama_server_tokens_predicted_total",
    "Total tokens predicted by the server",
    ["server_id", "model_id"],
    registry=_registry,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Instrument every request with Prometheus counters and histograms."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        endpoint = request.url.path

        active_http_connections.inc()
        start = time.monotonic()
        status = 500

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.monotonic() - start
            http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            active_http_connections.dec()

        return response


def get_metrics_registry() -> CollectorRegistry:
    return _registry
