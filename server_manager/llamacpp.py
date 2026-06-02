"""
LlamaCppServerManager - Specialized server manager for llama.cpp servers.
"""

from typing import List, Optional

from models import Model, ModelTask
from server_manager.base import BaseServerManager
from server_manager.llamacpp_argument_builder import LlamaCppArgumentBuilder


class LlamaCppServerManager(BaseServerManager):
    """Manages llama.cpp server process lifecycle."""

    def __init__(
        self,
        model: Model,
        session_id: Optional[str] = None,
        port: Optional[int] = None,
        is_embedding: Optional[bool] = None,
    ):
        super().__init__(
            model=model,
            session_id=session_id,
            port=port,
            startup_timeout=120,
        )
        # Embedding mode is derived from the model's task so any model
        # declared `task: TextToEmbeddings` is automatically served with
        # llama.cpp's --embedding flag. The router constructs this
        # manager without passing is_embedding, so without this the
        # embedding path was never reached. An explicit arg still
        # overrides for callers that need it.
        if is_embedding is None:
            self.is_embedding = (
                getattr(model, "task", None) == ModelTask.TEXTTOEMBEDDINGS
            )
        else:
            self.is_embedding = is_embedding

    def get_api_endpoint(self, path: str) -> str:
        """Get the full URL for a specific API endpoint."""
        if path in ["/health", "/metrics"]:
            return f"{self.server_url}{path}"
        else:
            return f"{self.server_url}/v1{path}"

    def _build_server_args(self) -> List[str]:
        """Build command line arguments for llama.cpp server."""
        builder = LlamaCppArgumentBuilder(
            model=self.model,
            port=self.port,
            is_embedding=self.is_embedding,
        )
        args = builder.build_args()
        self._logger.info(f"Server args: {' '.join(args)}")
        return args
