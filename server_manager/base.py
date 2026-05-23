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
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import requests

from models import Model
from utils.logging import llmmllogger

# Number of recent stderr/stdout lines to retain for crash diagnostics.
_STDERR_RING_BUFFER_SIZE = 200
# Number of those lines to dump when the child process dies unexpectedly.
_STDERR_DUMP_ON_DEATH = 50


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
        )
        self.process: Optional[subprocess.Popen] = None
        self.port: int = port or self._find_available_port()

        # FIX: Use 127.0.0.1 explicitly to match llama-server host and avoid IPv6/localhost ambiguity
        self.server_url = f"http://127.0.0.1:{self.port}"  # No /v1 suffix

        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.startup_timeout = startup_timeout
        self.pid: Optional[int] = None

        # ------------------------------------------------------------------
        # Crash / unexpected-exit instrumentation.
        # ------------------------------------------------------------------
        # Ring buffer of recent stderr lines, so when the child dies we can
        # dump the *actual* error (e.g. GGML_ASSERT) instead of just an exit
        # code.  Protected implicitly by the GIL — deque append/pop is atomic.
        self._stderr_buffer: Deque[str] = deque(maxlen=_STDERR_RING_BUFFER_SIZE)

        # Flag set by stop() (or any other intentional shutdown path) so the
        # watchdog can distinguish "we asked the process to exit" from "it
        # crashed on its own".
        self._intentional_stop: bool = False

        # Watchdog thread that awaits proc.wait() and purges the cache entry
        # on exit.  Created in _spawn_watchdog() after the process is alive.
        self._watchdog_thread: Optional[threading.Thread] = None

        # Cache hook — populated via attach_lifecycle() so the watchdog can
        # call ServerCache.purge_dead_server(server_id) without an import
        # cycle.
        self._server_id: Optional[str] = None
        self._cache_ref: Any = None  # ServerCache, avoid import cycle

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

    # ------------------------------------------------------------------
    # Lifecycle hooks for ServerCache integration.
    # ------------------------------------------------------------------

    def attach_lifecycle(self, server_id: str, cache: Any) -> None:
        """Tell the manager which cache entry it belongs to.

        The ServerCache calls this from register_starting() / register() so
        that, on unexpected process death, the watchdog thread can purge the
        stale entry from the cache immediately (instead of waiting for the
        idle-eviction reaper to notice).

        Safe to call multiple times.  Has no effect on an already-running
        watchdog — wiring is just used at exit time.
        """
        self._server_id = server_id
        self._cache_ref = cache

    def _spawn_watchdog(self) -> None:
        """Start the per-process watchdog thread.

        Called from start() / _retry_with_reduced_context() once the
        subprocess is alive.  Idempotent — replaces any prior thread that
        belonged to an earlier process.
        """
        proc = self.process
        if proc is None:
            return

        # If a previous watchdog is still alive (e.g. after retry), let it
        # die naturally — its target process has already exited so wait()
        # returned, and the thread will exit on its own.
        thread = threading.Thread(
            target=self._watchdog_run,
            args=(proc,),
            name=f"server-watchdog-{self.port}",
            daemon=True,
        )
        self._watchdog_thread = thread
        thread.start()

    def _watchdog_run(self, proc: subprocess.Popen) -> None:
        """Block on proc.wait(); on exit, log + purge cache + record metric."""
        try:
            returncode = proc.wait()
        except Exception as e:
            self._logger.warning(f"Watchdog wait() failed: {e}")
            return

        # Tail of stderr for crash diagnostics.  Snapshot first so we don't
        # race the drainer threads if the process exits with output still
        # buffered.
        tail = list(self._stderr_buffer)[-_STDERR_DUMP_ON_DEATH:]
        tail_text = "\n".join(tail) if tail else "<no captured output>"

        intentional = self._intentional_stop
        # SIGTERM (143 / -15) is what stop() sends; treat it as intentional
        # even if the flag wasn't set (e.g. external `kill` from k8s).
        is_sigterm = returncode in (0, -15, 143)

        if intentional or is_sigterm:
            self._logger.info(
                f"Server process exited cleanly (exit code: {returncode}, "
                f"intentional={intentional})"
            )
        else:
            self._logger.warning(
                f"Server process died unexpectedly (exit code: {returncode}, "
                f"pid: {self.pid}). Last {len(tail)} captured output lines:\n"
                f"{tail_text}"
            )

        # Purge the cache entry ONLY when the process died unexpectedly.
        # The intentional path (stop(), evict_idle(), _retry_with_reduced_
        # context's kill of the old process) is responsible for its own
        # cache bookkeeping — and crucially, _retry_with_reduced_context
        # reuses the same server_id for the replacement process, so a
        # stale watchdog firing purge here would orphan the new process.
        if not (intentional or is_sigterm):
            if self._cache_ref is not None and self._server_id is not None:
                try:
                    self._cache_ref.purge_dead_server(self._server_id)
                except Exception as e:
                    self._logger.warning(
                        f"Failed to purge server {self._server_id} from cache: {e}"
                    )

        # Prometheus: distinguish unexpected death from intentional stops.
        if not (intentional or is_sigterm):
            try:
                # Import lazily to avoid pulling Prometheus into module init
                # for tests / non-runner contexts.
                from middleware.runner_metrics import record_server_eviction

                record_server_eviction("process_died")
            except Exception as e:
                self._logger.debug(f"Could not record process_died metric: {e}")

    @abstractmethod
    def get_api_endpoint(self, path: str) -> str:
        """Get the full URL for a specific API endpoint."""

    def _build_subprocess_env(self) -> Optional[Dict[str, str]]:
        """Build the environment dict passed to the child server process.

        Default: ``None`` — child inherits the runner's environment.
        Subclasses override when they need to scope a variable (e.g.
        ``CUDA_VISIBLE_DEVICES`` to confine sd-server to a single GPU).
        Returning ``None`` keeps the inheritance path so we don't have
        to enumerate the parent env explicitly.
        """
        return None

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

                # Start the process and capture output so we can log failures.
                # ``_build_subprocess_env`` is a hook for subclasses that need
                # to mutate the child env — e.g. SD pins CUDA_VISIBLE_DEVICES
                # so it lands on a specific GPU instead of always CUDA:0.
                env = self._build_subprocess_env()
                self.process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    env=env,
                )

                # Log PID
                self._logger.info(
                    f"Process started with PID {self.process.pid}, waiting for server readiness..."
                )
                self.pid = self.process.pid

                # Reset state for a fresh process so stale stderr from a
                # prior crash / retry doesn't contaminate the next watchdog.
                self._intentional_stop = False
                self._stderr_buffer.clear()

                # Stream subprocess stdout/stderr into our logger so failures
                # are visible, AND tee stderr into the ring buffer so the
                # watchdog can dump it on unexpected exit.
                if self.process.stdout:
                    t_out = threading.Thread(
                        target=self._stream_pipe,
                        args=(self.process.stdout, self._logger.debug, False),
                        daemon=True,
                    )
                    t_out.start()

                if self.process.stderr:
                    t_err = threading.Thread(
                        target=self._stream_pipe,
                        args=(self.process.stderr, self._logger.info, True),
                        daemon=True,
                    )
                    t_err.start()

                # Spawn the death watchdog as soon as the process exists, so
                # we catch crashes that happen during startup as well as
                # crashes that happen long after startup succeeded.
                self._spawn_watchdog()

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

    def _stream_pipe(self, pipe, log_fn, is_stderr: bool) -> None:
        """Drain a subprocess pipe line-by-line.

        Each line is logged via ``log_fn`` and, if it came from stderr,
        also appended to the ring buffer that the watchdog dumps on
        unexpected exit.  We log stderr at INFO (not DEBUG) because that is
        where llama.cpp emits GGML_ASSERT / CUDA errors; losing those is
        what motivated this change.

        Also attempts to attribute each line to a session_id by parsing
        the llama.cpp slot id (when present) and looking up the proxy's
        SlotLRU.  Helps correlate per-slot llama.cpp internals (prompt
        progress, checkpoint creation, slot launches) to the user
        session that owns the slot.  Best-effort: lines without a slot
        reference, or boot-time logs before any session is pinned,
        simply omit the field.
        """
        try:
            if not pipe:
                return
            # Late-import to avoid module-load cycles; this only happens
            # once per drain thread, not per line.
            try:
                from proxy.router import (
                    session_for_slot,
                    slot_id_from_llamacpp_line,
                )
            except Exception:
                session_for_slot = None  # type: ignore[assignment]
                slot_id_from_llamacpp_line = None  # type: ignore[assignment]

            server_id = getattr(self, "_server_id", "") or ""

            with pipe:
                # Compiled here once per drain thread instead of per line.
                #
                # Two patterns from llama.cpp's prompt-processing stage:
                #
                # (a) ``n_past = X, slot.prompt.tokens.size() = Y`` — fires
                #     only on the WARN branch (e.g. cache_reuse rejection,
                #     checkpoint erase).  Gives us the cached prefix size
                #     directly: X cached / Y total.
                #
                # (b) ``prompt processing, n_tokens = X, progress = 1.00,
                #     t = Y s / Z tokens per second`` — fires on the INFO
                #     branch at the END of prompt processing for every turn.
                #     Gives us the tokens processed (i.e. NOT cached) in
                #     this turn plus the prefill wall-time.  Combined with
                #     the body-size fingerprint on the API side we can
                #     reconstruct the cache-hit ratio on every turn.
                import re as _re_local
                _n_past_re = _re_local.compile(
                    r"task (\d+) \| n_past = (\d+), slot\.prompt\.tokens\.size\(\) = (\d+)"
                )
                _print_timing_re = _re_local.compile(
                    r"task (\d+) \| prompt processing, n_tokens = +(\d+), "
                    r"progress = 1\.00, t = +([\d.]+) s / +([\d.]+) tokens"
                )

                for line in iter(pipe.readline, ""):
                    if not line:
                        continue
                    stripped = line.rstrip()
                    if is_stderr:
                        # Bounded — deque(maxlen=...) drops oldest.
                        self._stderr_buffer.append(stripped)
                    extra_kwargs = {}
                    if (
                        server_id
                        and slot_id_from_llamacpp_line is not None
                        and session_for_slot is not None
                    ):
                        try:
                            sid = slot_id_from_llamacpp_line(stripped)
                            if sid is not None:
                                resolved = session_for_slot(server_id, sid)
                                if resolved:
                                    extra_kwargs["session_id"] = resolved
                                    extra_kwargs["slot_id"] = sid

                                    # Pattern (a) — warn branch with full
                                    # n_past + n_prompt.  When it fires we
                                    # can compute hit_pct directly.
                                    m = _n_past_re.search(stripped)
                                    if m:
                                        try:
                                            task_id = int(m.group(1))
                                            n_past = int(m.group(2))
                                            n_prompt = int(m.group(3))
                                            hit_pct = (
                                                (100.0 * n_past / n_prompt)
                                                if n_prompt
                                                else 0.0
                                            )
                                            log_fn(
                                                "Slot cache-hit per-turn",
                                                session_id=resolved,
                                                slot_id=sid,
                                                task_id=task_id,
                                                n_past=n_past,
                                                n_prompt=n_prompt,
                                                cache_hit_pct=round(hit_pct, 2),
                                                source="n_past_warn",
                                            )
                                        except Exception:
                                            pass

                                    # Pattern (b) — info branch, end of
                                    # prompt processing.  Fires once per
                                    # turn.  Gives us tokens processed
                                    # (NOT cached) + prefill wall-time.
                                    # Pair with the API-side fingerprint
                                    # body_bytes to compute approximate
                                    # cache hit rate.
                                    m = _print_timing_re.search(stripped)
                                    if m:
                                        try:
                                            task_id = int(m.group(1))
                                            n_processed = int(m.group(2))
                                            t_seconds = float(m.group(3))
                                            tok_per_sec = float(m.group(4))
                                            log_fn(
                                                "Slot prefill complete",
                                                session_id=resolved,
                                                slot_id=sid,
                                                task_id=task_id,
                                                n_processed=n_processed,
                                                prefill_seconds=round(
                                                    t_seconds, 2
                                                ),
                                                prefill_tokens_per_sec=round(
                                                    tok_per_sec, 2
                                                ),
                                            )
                                        except Exception:
                                            pass
                        except Exception:
                            # Best-effort enrichment; never break the drain.
                            pass
                    try:
                        log_fn(stripped, **extra_kwargs)
                    except TypeError:
                        # log_fn doesn't accept kwargs (some loggers); retry
                        # without enrichment.
                        try:
                            log_fn(stripped)
                        except Exception:
                            pass
                    except Exception:
                        # Logging must never kill the drain thread.
                        pass
        except Exception:
            # best-effort - avoid crashing background drainers
            pass

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

            # Kill the crashed process.  Mark this kill as intentional so the
            # previous process's watchdog doesn't trigger a spurious
            # "process_died" purge / metric on us.
            if self.process:
                try:
                    self._intentional_stop = True
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
                env=self._build_subprocess_env(),
            )
            self.pid = self.process.pid

            # Reset for fresh process — clear stale buffer + intentional flag
            # so the new watchdog starts clean.
            self._intentional_stop = False
            self._stderr_buffer.clear()

            if self.process.stdout:
                t_out = threading.Thread(
                    target=self._stream_pipe,
                    args=(self.process.stdout, self._logger.debug, False),
                    daemon=True,
                )
                t_out.start()

            if self.process.stderr:
                t_err = threading.Thread(
                    target=self._stream_pipe,
                    args=(self.process.stderr, self._logger.info, True),
                    daemon=True,
                )
                t_err.start()

            # New process → new watchdog.
            self._spawn_watchdog()

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

                # Tell the watchdog this exit is expected, BEFORE we send
                # any signal — otherwise the watchdog races us, sees the
                # SIGTERM exit code, and (defensively) logs a "died
                # unexpectedly" warning + emits a process_died metric.
                self._intentional_stop = True

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
