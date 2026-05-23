"""SDCppServerManager — manages a stable-diffusion.cpp ``sd-server`` process.

Shares the lifecycle / watchdog / log-drain machinery of
:class:`BaseServerManager` with the llama.cpp variant; the differences are:

  * The binary is ``sd-server`` (path from ``SD_SERVER_EXECUTABLE``).
  * There is no ``/health`` endpoint, so readiness is probed via
    ``GET /sdcpp/v1/capabilities`` which returns 200 once the model
    weights are loaded.
  * There is no context-size concept, so the context-validation step
    that ``BaseServerManager`` calls is a no-op (we override
    ``_validate_context_size``).
  * SIGSEGV-on-OOM retry-with-smaller-context doesn't make sense for
    image diffusion — instead, SIGSEGV during startup means the GGUF /
    VAE paths are wrong or the GPU couldn't fit the model.  We let
    BaseServerManager's retry kick in (it inspects ``--ctx-size``),
    which finds no match and falls through to normal failure handling.

The proxy router in ``proxy/router.py`` is provider-agnostic: it
forwards ``/v1/server/<id>/...`` to ``http://127.0.0.1:<port>/...``
unchanged, so callers reach sd-server endpoints (e.g.
``/sdapi/v1/txt2img``) via the same mechanism.
"""

from typing import List, Optional

import requests

from models import Model
from server_manager.base import BaseServerManager
from server_manager.sd_cpp_argument_builder import SDCppArgumentBuilder


class SDCppServerManager(BaseServerManager):
    """Manage the lifecycle of a stable-diffusion.cpp ``sd-server`` process."""

    def __init__(
        self,
        model: Model,
        session_id: Optional[str] = None,
        port: Optional[int] = None,
    ):
        # Image diffusion models load slower than LLMs (multiple weight
        # files to mmap, plus VAE / text encoder initialisation), so we
        # bump the startup timeout above the llama.cpp default.
        super().__init__(
            model=model,
            session_id=session_id,
            port=port,
            startup_timeout=300,
        )

    # ------------------------------------------------------------------
    # Endpoint plumbing
    # ------------------------------------------------------------------

    def get_api_endpoint(self, path: str) -> str:
        # sd-server does not expose ``/health``; map it onto
        # ``/sdcpp/v1/capabilities`` which is the cheapest 200-returning
        # endpoint that proves the model finished loading.
        if path == "/health":
            return f"{self.server_url}/sdcpp/v1/capabilities"
        if path == "/metrics":
            # No native metrics endpoint — return /sdcpp/v1/capabilities
            # so polling code gets a 200 instead of a hang.  The caller
            # treats an empty body as "no metrics", which is correct.
            return f"{self.server_url}/sdcpp/v1/capabilities"
        # All sd-server routes are absolute (e.g. ``/sdapi/v1/txt2img``,
        # ``/v1/images/generations``) — don't inject a ``/v1`` prefix the
        # way the llama.cpp manager does.
        return f"{self.server_url}{path}"

    # ------------------------------------------------------------------
    # CLI args
    # ------------------------------------------------------------------

    def _build_server_args(self) -> List[str]:
        args = SDCppArgumentBuilder(model=self.model, port=self.port).build_args()
        self._logger.info(f"sd-server args: {' '.join(args)}")
        return args

    # ------------------------------------------------------------------
    # Overrides — features that don't apply to SD models
    # ------------------------------------------------------------------

    def _validate_context_size(self) -> bool:
        # sd-server has no context window; nothing to validate.  Returning
        # True keeps BaseServerManager.start() on the happy path.
        return True

    def is_running(self) -> bool:
        """Mirror BaseServerManager.is_running but use ``/sdcpp/v1/capabilities``."""
        if not self.process or self.process.poll() is not None:
            return False
        try:
            response = requests.get(self.get_api_endpoint("/health"), timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
