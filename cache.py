"""
Server cache for llmmllab-runner.

Manages registry of active llama.cpp server instances with use-count tracking
and two-tier idle eviction:

  - CACHE_TIMEOUT_MIN (soft):  server becomes *eligible* for eviction when
    idle for this long.  Used when VRAM pressure requires freeing space.
  - EVICTION_TIMEOUT_MIN (hard): server *must* be evicted once idle for this
    long, regardless of VRAM pressure.

Both timers start when the last client releases the server (use_count drops to 0)
and reset when a new client acquires it (use_count goes above 0).
"""

import threading
import time
import uuid
from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from config import CACHE_TIMEOUT_MIN, EVICTION_TIMEOUT_MIN

from server_manager import BaseServerManager
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ServerCache")


class ServerEntry(BaseModel):
    server_id: Annotated[str, Field(..., description="Unique server ID")]
    model_id: Annotated[str, Field(..., description="Model ID")]
    port: Annotated[int, Field(..., description="Port number")]
    use_count: Annotated[int, Field(default=0)] = 0
    created_at: Annotated[float, Field(default=0)] = 0
    healthy: Annotated[bool, Field(default=True)] = True
    starting: Annotated[bool, Field(default=False)] = False
    manager: Annotated[Optional[BaseServerManager], Field(default=None)] = (
        None  # BaseServerManager instance
    )
    # Set to time.time() when use_count drops to 0; None while in use.
    idle_since: Annotated[Optional[float], Field(default=None)] = None

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,  # for manager field
    )


class ServerCache:
    """Thread-safe registry of active llama.cpp servers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._servers: Dict[str, ServerEntry] = {}

    def register_starting(
        self,
        model_id: str,
        port: int,
        manager: Optional[BaseServerManager] = None,
    ) -> str:
        """Register a server that is starting (not yet healthy).

        Returns server_id. Caller must call mark_ready() once the server is up,
        or remove() if it fails.
        """
        server_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._servers[server_id] = ServerEntry(
                server_id=server_id,
                model_id=model_id,
                port=port,
                manager=manager,
                created_at=time.time(),
                starting=True,
            )
        logger.info(
            f"Registered starting server {server_id} for model {model_id} on port {port}"
        )
        return server_id

    def mark_ready(self, server_id: str) -> bool:
        """Mark a starting server as ready."""
        with self._lock:
            entry = self._servers.get(server_id)
            if not entry:
                return False
            entry.starting = False
        logger.info(f"Server {server_id} ready on port {entry.port}")
        return True

    def register(
        self,
        model_id: str,
        port: int,
        manager: Optional[BaseServerManager] = None,
    ) -> str:
        """Register a new server and return its server_id."""
        server_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._servers[server_id] = ServerEntry(
                server_id=server_id,
                model_id=model_id,
                port=port,
                manager=manager,
                created_at=time.time(),
            )
        logger.info(
            f"Registered server {server_id} for model {model_id} on port {port}"
        )
        return server_id

    def get(self, server_id: str) -> Optional[ServerEntry]:
        """Get a server entry by ID."""
        with self._lock:
            return self._servers.get(server_id)

    def acquire_by_model(
        self, model_id: str, log_miss: bool = True
    ) -> Optional[ServerEntry]:
        """Find an existing healthy server for the given model.

        Returns the server entry if found and healthy, None otherwise.
        Removes dead (unhealthy) servers from the cache.
        Does NOT increment use_count — activity tracking is handled by the
        proxy router which increments/decrements on each proxied request.
        """
        eligible = None
        dead_ids = []

        with self._lock:
            for entry in self._servers.values():
                if entry.model_id != model_id:
                    continue
                if entry.starting:
                    # Server is still starting — treat as "exists" so we don't
                    # create a duplicate. Return None to signal "try again later".
                    continue
                if entry.manager is not None and hasattr(entry.manager, "is_running"):
                    if not entry.manager.is_running():
                        dead_ids.append(entry.server_id)
                        continue
                eligible = entry

            # Clean up dead servers
            for sid in dead_ids:
                del self._servers[sid]

            if eligible is not None:
                eligible.idle_since = None

        if eligible:
            logger.info(
                f"Acquired existing server {eligible.server_id} "
                f"for model {model_id}"
            )
        elif log_miss:
            logger.debug(f"No ready server for model {model_id}")
        return eligible

    def has_starting_server(self, model_id: str) -> bool:
        """Check if there's a server starting for this model."""
        with self._lock:
            return any(
                e.model_id == model_id and e.starting for e in self._servers.values()
            )

    def increment_use(self, server_id: str) -> bool:
        """Increment use count and clear idle timer. Returns False if not found."""
        with self._lock:
            entry = self._servers.get(server_id)
            if not entry:
                return False
            entry.use_count += 1
            entry.idle_since = None
        return True

    def decrement_use(self, server_id: str) -> bool:
        """Decrement use count. Sets idle_since when count drops to 0."""
        with self._lock:
            entry = self._servers.get(server_id)
            if not entry:
                return False
            entry.use_count = max(0, entry.use_count - 1)
            if entry.use_count == 0:
                entry.idle_since = time.time()
        return True

    # ------------------------------------------------------------------
    # Soft eviction — returns servers eligible for eviction under VRAM
    # pressure (idle longer than CACHE_TIMEOUT_MIN).
    # ------------------------------------------------------------------

    def get_eligible_for_eviction(self) -> List[ServerEntry]:
        """Return idle servers that exceed the soft cache timeout.

        These servers *may* be evicted when VRAM pressure requires space for a
        new model.  Callers are responsible for actually stopping and removing
        them.  Does NOT mutate the cache.
        """
        cutoff = time.time() - CACHE_TIMEOUT_MIN * 60
        eligible = []
        with self._lock:
            for entry in self._servers.values():
                if entry.idle_since is not None and entry.idle_since <= cutoff:
                    eligible.append(entry)
        return eligible

    # ------------------------------------------------------------------
    # Hard eviction — removes servers that exceed EVICTION_TIMEOUT_MIN.
    # ------------------------------------------------------------------

    def evict_idle(self) -> List[str]:
        """Stop and remove servers that have been idle beyond the hard eviction timeout.

        Actually stops the llama-server process via the manager, then removes
        the entry from the cache.  Returns list of evicted server_ids.
        """
        cutoff = time.time() - EVICTION_TIMEOUT_MIN * 60
        evicted = []
        with self._lock:
            for server_id, entry in list(self._servers.items()):
                if entry.idle_since is not None and entry.idle_since <= cutoff:
                    evicted.append(entry)
                    del self._servers[server_id]
        for entry in evicted:
            if entry.manager is not None:
                try:
                    entry.manager.stop()
                    logger.info(
                        f"Hard-evicted idle server {entry.server_id} "
                        f"(model {entry.model_id}, port {entry.port})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error stopping evicted server {entry.server_id}: {e}"
                    )
            else:
                logger.info(f"Hard-evicted idle server {entry.server_id} (no manager)")
        return [e.server_id for e in evicted]

    def remove(self, server_id: str) -> Optional[ServerEntry]:
        """Remove a server from the cache. Returns the entry if it existed."""
        with self._lock:
            entry = self._servers.pop(server_id, None)
        if entry:
            # Stop manager for starting servers that never became healthy
            if entry.starting and entry.manager is not None:
                try:
                    entry.manager.stop()
                except Exception:
                    pass
            logger.info(f"Removed server {server_id}")
        return entry

    def stats(self) -> Dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "active_servers": len(self._servers),
                "servers": [
                    {
                        "server_id": e.server_id,
                        "model_id": e.model_id,
                        "port": e.port,
                        "use_count": e.use_count,
                        "healthy": e.healthy,
                        "starting": e.starting,
                        "created_at": e.created_at,
                        "idle_since": e.idle_since,
                    }
                    for e in self._servers.values()
                ],
            }

    def stop_all(self) -> None:
        """Stop all managed servers."""
        with self._lock:
            entries = list(self._servers.values())
        for entry in entries:
            if entry.manager is not None:
                try:
                    logger.info(f"Stopping server {entry.server_id}")
                    entry.manager.stop()
                except Exception as e:
                    logger.error(f"Error stopping server {entry.server_id}: {e}")
        with self._lock:
            self._servers.clear()
