"""
Base ServerManager - Common functionality for any server process management.

This provides a foundation for managing server processes regardless of the
underlying implementation (llama.cpp, vLLM, etc.).
"""

import json
import socket
import subprocess
import threading
import time
import traceback as tb_module
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

from models import Model
from utils.logging import llmmllogger


class BaseServerManager(ABC):
    """Abstract base class for managing server processes."""

    def __init__(
        self,
        model: Model,
        session_id: Optional[str] = None,
        port: Optional[int] = None,
        startup_timeout: int = 30,
    ):
        self.model = model
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__,
            model=model.name,
            session=session_id or "N/A",
        )
        self.process: Optional[subprocess.Popen] = None
        self.port: int = port or self._find_available_port()

        # FIX: Use 127.0.0.1 explicitly to match llama-server host and avoid IPv6/localhost ambiguity
        self.server_url = f"http://127.0.0.1:{self.port}"  # No /v1 suffix

        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.startup_timeout = startup_timeout
        self.pid: Optional[int] = None

    def _find_available_port(self, start_port: int = 8001) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # FIX: Ensure we don't accidentally reuse a port in TIME_WAIT
                # by NOT setting SO_REUSEADDR. This forces us to skip ports
                # that are not fully clean.
                # Also bind explicitly to 127.0.0.1 to match server config.
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports found")

    @abstractmethod
    def _build_server_args(self) -> List[str]:
        """Build command line arguments for the server."""

    @abstractmethod
    def get_api_endpoint(self, path: str) -> str:
        """Get the full URL for a specific API endpoint."""

    def start(self) -> bool:
        """Start the server process."""
        with self._lock:
            # DEBUG: Add detailed server start logging
            call_stack = tb_module.extract_stack()[-4:-1]
            call_info = " -> ".join(
                [
                    f"{frame.filename.split('/')[-1]}:{frame.lineno}"
                    for frame in call_stack
                ]
            )

            if self.process and self.process.poll() is None:
                self._logger.info(
                    f"Server already running on port {self.port} (called from {call_info})"
                )
                return True

            self._logger.info(
                f"Starting NEW server on port {self.port} (called from {call_info})"
            )

            try:
                args = self._build_server_args()

                self._logger.info(f"Starting server on port {self.port}")
                self._logger.debug(f"Command: {' '.join(args)}")

                # Start the process and capture output so we can log failures
                self.process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )

                # Log PID
                self._logger.info(
                    f"Process started with PID {self.process.pid}, waiting for server readiness..."
                )
                self.pid = self.process.pid

                # Stream subprocess stdout/stderr into our logger so failures are visible
                def _stream_pipe(pipe, log_fn):
                    try:
                        if not pipe:
                            return
                        with pipe:
                            for line in iter(pipe.readline, ""):
                                if line:
                                    log_fn(line.rstrip())
                    except Exception:
                        # best-effort - avoid crashing main thread
                        pass

                if self.process.stdout:
                    t_out = threading.Thread(
                        target=_stream_pipe,
                        args=(self.process.stdout, self._logger.debug),
                        daemon=True,
                    )
                    t_out.start()

                if self.process.stderr:
                    t_err = threading.Thread(
                        target=_stream_pipe,
                        args=(self.process.stderr, self._logger.debug),
                        daemon=True,
                    )
                    t_err.start()

                # Wait for server to be ready
                if self._wait_for_server():
                    self._logger.info(
                        f"Server started successfully on port {self.port}"
                    )
                    # Validate actual context size after boot — stop server
                    # if llama.cpp reduced context below the configured minimum.
                    if not self._validate_context_size():
                        self._logger.error(
                            "Context validation failed — stopping server"
                        )
                        self.stop()
                        return False
                    return True
                else:
                    # Check if the process died with SIGSEGV (exit code -11)
                    # This often indicates the model's num_ctx is too large for
                    # available memory. Retry with a reduced context window.
                    if self.process and self.process.returncode == -11:
                        return self._retry_with_reduced_context(args)
                    self._logger.error(
                        "Server failed to start within timeout - cleaning up"
                    )
                    return False

            except Exception as e:
                self._logger.error(f"Failed to start server: {e}")
                return False

    def _retry_with_reduced_context(self, original_args: List[str]) -> bool:
        """Retry server start with reduced num_ctx after SIGSEGV.

        When the server crashes with exit code -11 (SIGSEGV) during startup,
        it's often because the requested context window is too large for
        available memory. This method progressively reduces num_ctx and
        retries up to 3 times.

        Args:
            original_args: The original command line arguments.

        Returns:
            True if the server started successfully, False otherwise.
        """
        import re

        ctx_match = re.search(r"--ctx-size\s+(\d+)", " ".join(original_args))
        if not ctx_match:
            self._logger.warning(
                "Cannot reduce context: --ctx-size not found in server args"
            )
            return False

        original_ctx = int(ctx_match.group(1))
        self._logger.warning(
            f"Server crashed with SIGSEGV (exit code -11). "
            f"Retrying with reduced context window (original: {original_ctx})"
        )

        minimum_ctx = self._get_minimum_ctx()
        for attempt in range(1, 4):
            reduced_ctx = int(original_ctx * (0.5**attempt))
            reduced_ctx = max(
                reduced_ctx, minimum_ctx
            )  # Respect model-configured minimum

            retry_args = [
                a if not a.startswith("--ctx-size") else f"--ctx-size {reduced_ctx}"
                for a in original_args
            ]
            # Also replace the value after --ctx-size
            retry_args = []
            for i, a in enumerate(original_args):
                if a == "--ctx-size" and i + 1 < len(original_args):
                    retry_args.append("--ctx-size")
                    retry_args.append(str(reduced_ctx))
                else:
                    retry_args.append(a)

            self._logger.info(
                f"Retry attempt {attempt}/3 with num_ctx={reduced_ctx} "
                f"(reduced from {original_ctx})"
            )

            # Kill the crashed process
            if self.process:
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception:
                    pass

            self.process = subprocess.Popen(
                retry_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self.pid = self.process.pid

            # Stream output
            def _stream_pipe(pipe, log_fn):
                try:
                    if not pipe:
                        return
                    with pipe:
                        for line in iter(pipe.readline, ""):
                            if line:
                                log_fn(line.rstrip())
                except Exception:
                    pass

            if self.process.stdout:
                t_out = threading.Thread(
                    target=_stream_pipe,
                    args=(self.process.stdout, self._logger.debug),
                    daemon=True,
                )
                t_out.start()

            if self.process.stderr:
                t_err = threading.Thread(
                    target=_stream_pipe,
                    args=(self.process.stderr, self._logger.debug),
                    daemon=True,
                )
                t_err.start()

            if self._wait_for_server():
                self._logger.info(
                    f"Server started successfully with reduced context "
                    f"num_ctx={reduced_ctx} (original: {original_ctx})"
                )
                if not self._validate_context_size():
                    self._logger.error(
                        "Context validation failed on retry — stopping server"
                    )
                    self.stop()
                    continue
                return True

            # If this retry also SIGSEGV, don't recurse — continue the loop
            if self.process and self.process.returncode == -11:
                self._logger.warning(
                    f"Retry attempt {attempt} also crashed with SIGSEGV, "
                    f"trying with even smaller context"
                )
                continue

            self._logger.error(
                f"Retry attempt {attempt} failed (exit code: "
                f"{self.process.returncode if self.process else 'unknown'})"
            )

        self._logger.error(
            f"All retry attempts failed for model {self.model.name}. "
            f"Server cannot start with any reduced context window. "
            f"Original num_ctx: {original_ctx}."
        )
        return False

    def _get_minimum_ctx(self) -> int:
        """Compute the minimum acceptable context size from model config.

        Uses ctx_size_reduction_limit (a fraction of num_ctx) to determine
        how far llama.cpp is allowed to auto-reduce the context window.
        Falls back to 2048 as an absolute floor.
        """
        configured_ctx = getattr(self.model.parameters, "num_ctx", None) or 90000
        reduction_limit = (
            getattr(self.model.parameters, "ctx_size_reduction_limit", None)
            if self.model.parameters
            else None
        )
        if reduction_limit is None:
            reduction_limit = 0.5
        import math

        return max(math.ceil(configured_ctx * reduction_limit), 2048)

    def _validate_context_size(self) -> bool:
        """Check the actual context size after server boot.

        Returns True if context is acceptable, False if it was reduced
        below the configured minimum (ctx_size_reduction_limit * num_ctx).
        When context is below minimum, the server is stopped and start() fails.
        """
        try:
            models_endpoint = self.get_api_endpoint("/v1/models")
            resp = requests.get(models_endpoint, timeout=5)
            if resp.status_code == 200:
                models = resp.json()
                if models:
                    model_info = models[0]
                    actual_ctx = (
                        model_info.get("max_model_len")
                        or model_info.get("max_context")
                        or model_info.get("context_length")
                    )
                    if actual_ctx is not None:
                        configured_ctx = (
                            getattr(self.model.parameters, "num_ctx", None) or 90000
                        )
                        actual_ctx = int(actual_ctx)
                        minimum_ctx = self._get_minimum_ctx()

                        if actual_ctx < minimum_ctx:
                            self._logger.error(
                                f"Context size reduced below minimum: "
                                f"configured={configured_ctx}, actual={actual_ctx}, "
                                f"minimum={minimum_ctx} ({actual_ctx/configured_ctx*100:.0f}% of requested). "
                                f"Server cannot serve reliably — shutting down. "
                                f"Reduce num_ctx in model config or increase GPU VRAM."
                            )
                            return False
                        elif actual_ctx < configured_ctx:
                            self._logger.warning(
                                f"Context size reduced by server (within acceptable range): "
                                f"configured={configured_ctx}, actual={actual_ctx} "
                                f"({actual_ctx/configured_ctx*100:.0f}% of requested, minimum={minimum_ctx}). "
                                f"Conversations may fail when exceeding {actual_ctx} tokens."
                            )
                        else:
                            self._logger.debug(
                                f"Context size OK: configured={configured_ctx}, actual={actual_ctx}"
                            )
        except Exception as e:
            self._logger.debug(f"Could not validate context size: {e}")
        return True

    def _wait_for_server(self) -> bool:
        """Wait for server to become ready."""
        health_endpoint = self.get_api_endpoint("/health")

        self._logger.debug(
            f"Waiting for server health at {health_endpoint} for up to {self.startup_timeout} seconds"
        )

        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            # Check if process is still alive
            if self.process and self.process.poll() is not None:
                # Process exited early - capture any remaining output for debugging
                try:
                    out, err = self.process.communicate(timeout=1)
                    if out:
                        self._logger.error(f"Server stdout before exit:\n{out}")
                    if err:
                        self._logger.error(f"Server stderr before exit:\n{err}")
                except Exception:
                    # ignore communicate errors
                    pass

                self._logger.error(
                    f"Server process died unexpectedly (exit code: {self.process.returncode})"
                )
                return False

            try:
                response = requests.get(
                    health_endpoint, timeout=3
                )  # Increased timeout to 3 seconds
                if response.status_code == 200:
                    self._logger.info(
                        f"Server and model ready - /health responding {response.status_code}"
                    )
                    return True
                else:
                    # For some servers, 503 might indicate "still loading"
                    if response.status_code == 503:
                        self._logger.debug(
                            f"Model still loading - /health returns 503: {response.text}"
                        )
                    else:
                        self._logger.debug(
                            f"Health check returned {response.status_code}: {response.text}"
                        )
            except requests.exceptions.RequestException:
                pass  # Server not ready yet

            time.sleep(1.5)  # Slightly longer sleep interval

        # If we get here, timeout was reached - ensure process cleanup
        self._logger.error(
            f"Server startup timed out after {self.startup_timeout} seconds"
        )
        return False

    def stop(self) -> bool:
        """Stop the server process."""
        with self._lock:
            if not self.process:
                return True

            try:
                self._logger.info(f"Stopping server on port {self.port}")

                # Send SIGTERM for graceful shutdown
                self.process.terminate()

                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._logger.warning(
                        "Server didn't shut down gracefully, force killing"
                    )
                    self.process.kill()
                    self.process.wait()

                self.process = None
                self._logger.info("Server stopped successfully")
                return True

            except Exception as e:
                self._logger.error(f"Error stopping server: {e}")
                return False

    def is_running(self) -> bool:
        """Check if server is running and responsive."""
        if not self.process or self.process.poll() is not None:
            return False

        try:
            health_endpoint = self.get_api_endpoint("/health")
            response = requests.get(health_endpoint, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        try:
            metrics_endpoint = self.get_api_endpoint("/metrics")
            response = requests.get(metrics_endpoint, timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}

    def __del__(self):
        """Cleanup when server manager is destroyed."""
        try:
            self.stop()
        except Exception as e:
            self._logger.warning(f"Error during server manager cleanup: {e}")

        del self.process
