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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import CACHE_TIMEOUT_MIN, EVICTION_TIMEOUT_MIN

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ServerCache")


@dataclass
class _ServerEntry:
    server_id: str
    model_id: str
    port: int
    use_count: int = 0
    created_at: float = field(default_factory=time.time)
    healthy: bool = True
    manager: object = None  # LlamaCppServerManager instance
    # Set to time.time() when use_count drops to 0; None while in use.
    idle_since: Optional[float] = None


class ServerCache:
    """Thread-safe registry of active llama.cpp servers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._servers: Dict[str, _ServerEntry] = {}

    def register(
        self,
        model_id: str,
        port: int,
        manager: object = None,
    ) -> str:
        """Register a new server and return its server_id."""
        server_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._servers[server_id] = _ServerEntry(
                server_id=server_id,
                model_id=model_id,
                port=port,
                manager=manager,
            )
        logger.info(f"Registered server {server_id} for model {model_id} on port {port}")
        return server_id

    def get(self, server_id: str) -> Optional[_ServerEntry]:
        """Get a server entry by ID."""
        with self._lock:
            return self._servers.get(server_id)

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

    def get_eligible_for_eviction(self) -> List[_ServerEntry]:
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
        """Remove servers that have been idle beyond the hard eviction timeout.

        Returns list of evicted server_ids.
        """
        cutoff = time.time() - EVICTION_TIMEOUT_MIN * 60
        evicted = []
        with self._lock:
            for server_id, entry in list(self._servers.items()):
                if entry.idle_since is not None and entry.idle_since <= cutoff:
                    evicted.append(server_id)
                    del self._servers[server_id]
        for server_id in evicted:
            logger.info(f"Hard-evicted idle server {server_id}")
        return evicted

    def remove(self, server_id: str) -> Optional[_ServerEntry]:
        """Remove a server from the cache. Returns the entry if it existed."""
        with self._lock:
            entry = self._servers.pop(server_id, None)
        if entry:
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
