"""Abstract base class for in-process pipelines.

The runner supports two execution patterns:

1. **Subprocess servers** — llama.cpp and stable-diffusion.cpp ship their own
   HTTP servers; we just spawn them and proxy traffic.  These live in
   ``server_manager/`` and are routed through ``proxy/router.py``.

2. **In-process pipelines** — some models (TRELLIS, ComfyUI nodes, custom HF
   diffusers stacks) have no standalone server and must run inside the
   runner's Python process.  ``InProcessPipeline`` is the contract for those.

A pipeline is identified by ``name`` (e.g. ``"img23d"``), declares the task
family it serves (``ModelTask``), and exposes a single ``run(payload)``
coroutine that does the work.  The router under ``routers/pipelines.py``
calls ``run`` directly; there is no subprocess, no proxy, no HTTP hop.

Lazy initialisation:
  * ``load()`` is called the first time a request reaches the pipeline.
  * Subsequent requests reuse the loaded model.
  * ``unload()`` releases GPU memory; called by the runner on idle eviction
    (TODO once we wire pipeline eviction into the cache).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from models import ModelTask
from utils.logging import llmmllogger


class InProcessPipeline(ABC):
    """Abstract base class for an in-process inference pipeline."""

    #: Short stable identifier used in URL paths and the model registry.
    name: str = ""

    #: The :class:`ModelTask` this pipeline serves.  Used by the API layer
    #: to discover an appropriate pipeline for a given task.
    task: Optional[ModelTask] = None

    def __init__(self) -> None:
        self._logger = llmmllogger.bind(component=self.__class__.__name__)
        self._loaded = False
        self._load_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def _load(self) -> None:
        """Subclasses implement model loading here.

        Called exactly once before the first ``run`` invocation, and again
        after any explicit ``unload``.  Heavy GPU work belongs here.
        """

    @abstractmethod
    async def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Subclasses implement one inference call here.

        Receives the (already-validated) request payload, returns the
        response dict.  ``self._loaded`` is guaranteed to be ``True``.
        """

    async def unload(self) -> None:
        """Release GPU resources.  Subclasses may override; default no-op."""
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Public entry point — router calls this.
    # ------------------------------------------------------------------

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Load the model on first call, then delegate to :meth:`_run`.

        On exceptions (load or run), the full traceback is logged at
        ERROR level — without this, structlog's default formatter
        swallows the ``exc_info`` chain emitted by the surrounding
        router error handler, making CUDA OOM and similar failures
        nearly impossible to debug.

        After :meth:`_run` completes (whether successfully or with an
        error), if the env var ``IN_PROCESS_AUTO_UNLOAD`` is truthy,
        unload the model immediately — the image / 3D pipelines on
        this runner can be 6-20 GB resident and we don't keep them
        warm between requests by default once auto-unload is on.
        Subsequent requests pay the load cost again.
        """
        import os
        import traceback as _tb
        if not self._loaded:
            async with self._load_lock:
                # Re-check inside the lock — another coroutine may have
                # finished loading while we were waiting.
                if not self._loaded:
                    self._logger.info(f"Loading pipeline {self.name}")
                    try:
                        await self._load()
                    except Exception:
                        self._logger.error(
                            f"Pipeline {self.name} load failed:\n"
                            f"{_tb.format_exc()}"
                        )
                        raise
                    self._loaded = True
                    self._logger.info(f"Pipeline {self.name} ready")
        auto_unload = os.environ.get(
            "IN_PROCESS_AUTO_UNLOAD", ""
        ).lower() in ("1", "true", "yes", "on")
        try:
            return await self._run(payload)
        except Exception:
            self._logger.error(
                f"Pipeline {self.name} run failed:\n{_tb.format_exc()}"
            )
            raise
        finally:
            if auto_unload:
                try:
                    self._logger.info(
                        f"Auto-unloading pipeline {self.name} "
                        f"(IN_PROCESS_AUTO_UNLOAD=1); next request will "
                        f"pay the load cost again"
                    )
                    await self.unload()
                except Exception as e:  # noqa: BLE001
                    self._logger.warning(
                        f"Auto-unload of {self.name} failed: {e}"
                    )
