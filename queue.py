"""Priority queue for request scheduling in llmmllab-runner.

Provides a thread-safe, async-aware priority queue that orders server
creation requests by priority level and applies aging to prevent
starvation of low-priority requests.

Priority levels (lower number = higher priority):
  HIGH (1)  — Direct user engagement (X-Request-Priority: high)
  MEDIUM (2) — Scheduled tasks (X-Request-Priority: medium)
  LOW (3)   — Background / system tasks (X-Request-Priority: low)

Request sources:
  USER      — Authenticated user via API
  SCHEDULED — Cron jobs, scheduled maintenance
  SYSTEM    — Background / system maintenance

Aging policy:
  LOW  → MEDIUM  after 60 seconds in queue (default)
  MEDIUM → HIGH  after 120 seconds in queue (default)

Both thresholds are configurable via environment variables:
  QUEUE_AGING_LOW_TO_MEDIUM_SEC  (default: 60)
  QUEUE_AGING_MEDIUM_TO_HIGH_SEC (default: 120)
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Priority(Enum):
    """Request priority levels. Lower value = higher priority."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class RequestSource(Enum):
    """Origin of the request."""
    USER = "user"
    SCHEDULED = "scheduled"
    SYSTEM = "system"


# Mapping from header string values to Priority enum
PRIORITY_MAP = {
    "high": Priority.HIGH,
    "medium": Priority.MEDIUM,
    "low": Priority.LOW,
}

# Mapping from header string values to RequestSource enum
SOURCE_MAP = {
    "user": RequestSource.USER,
    "scheduled": RequestSource.SCHEDULED,
    "system": RequestSource.SYSTEM,
}

# Aging thresholds (seconds in queue before promotion)
# Configurable via environment variables.
AGING_LOW_TO_MEDIUM = int(os.environ.get("QUEUE_AGING_LOW_TO_MEDIUM_SEC", "60"))
AGING_MEDIUM_TO_HIGH = int(os.environ.get("QUEUE_AGING_MEDIUM_TO_HIGH_SEC", "120"))


@dataclass(order=True)
class QueuedRequest:
    """A request waiting in the priority queue.

    Ordering is by (effective_priority, enqueue_time) so that:
    - Higher priority requests are dequeued first
    - Within the same priority, FIFO order is preserved
    """
    sort_key: tuple = field(init=False)
    priority: Priority
    request: Any = field(compare=False)
    enqueue_time: float = field(default_factory=time.time)
    source: RequestSource = field(default=RequestSource.USER, compare=False)
    future: asyncio.Future = field(default=None, compare=False, repr=False)
    _original_priority: Priority = field(init=False, compare=False)

    def __post_init__(self):
        self._original_priority = self.priority
        self.sort_key = (self.priority.value, self.enqueue_time)

    def apply_aging(self) -> bool:
        """Apply aging policy. Returns True if priority was promoted."""
        elapsed = time.time() - self.enqueue_time
        if self.priority == Priority.LOW and elapsed >= AGING_LOW_TO_MEDIUM:
            self.priority = Priority.MEDIUM
            self.sort_key = (self.priority.value, self.enqueue_time)
            return True
        if self.priority == Priority.MEDIUM and elapsed >= AGING_MEDIUM_TO_HIGH:
            self.priority = Priority.HIGH
            self.sort_key = (self.priority.value, self.enqueue_time)
            return True
        return False


def parse_priority_header(value: Optional[str]) -> Priority:
    """Parse X-Request-Priority header value to Priority enum."""
    if value is None:
        return Priority.MEDIUM
    return PRIORITY_MAP.get(value.strip().lower(), Priority.MEDIUM)


def parse_source_header(value: Optional[str]) -> RequestSource:
    """Parse X-Request-Source header value to RequestSource enum."""
    if value is None:
        return RequestSource.USER
    return SOURCE_MAP.get(value.strip().lower(), RequestSource.USER)


class PriorityRequestQueue:
    """Async-safe priority queue for server creation requests.

    Uses a min-heap internally. All public methods are coroutine-safe
    and protected by an asyncio.Lock.
    """

    def __init__(self):
        self._heap: list[QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._counter = 0

    async def enqueue(
        self,
        request: Any,
        priority: Priority = Priority.MEDIUM,
        source: RequestSource = RequestSource.USER,
    ) -> asyncio.Future:
        """Add a request to the queue and return a Future to await.

        The Future resolves when the request is dequeued (its turn comes).
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        item = QueuedRequest(
            priority=priority,
            enqueue_time=time.time(),
            request=request,
            source=source,
            future=future,
        )

        async with self._lock:
            import heapq
            heapq.heappush(self._heap, item)

        return future

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Remove and return the highest-priority request.

        Applies aging to all items before selecting the next one,
        so long-waiting requests get promoted.
        """
        async with self._lock:
            if not self._heap:
                return None

            # Apply aging to all items
            for item in self._heap:
                item.apply_aging()

            import heapq
            item = heapq.heappop(self._heap)
            return item

    async def size(self) -> int:
        """Return current queue size."""
        async with self._lock:
            return len(self._heap)

    async def peek(self) -> Optional[QueuedRequest]:
        """Return the highest-priority item without removing it."""
        async with self._lock:
            if not self._heap:
                return None
            import heapq
            return self._heap[0]

    async def cancel(self, future: asyncio.Future) -> bool:
        """Remove a request from the queue by its Future.

        Returns True if the request was found and removed.
        """
        async with self._lock:
            for i, item in enumerate(self._heap):
                if item.future is future:
                    self._heap.pop(i)
                    import heapq
                    heapq.heapify(self._heap)
                    if not future.done():
                        future.cancel()
                    return True
        return False

    async def apply_aging_to_all(self):
        """Apply aging policy to all queued items (for background task)."""
        async with self._lock:
            for item in self._heap:
                item.apply_aging()
            import heapq
            heapq.heapify(self._heap)
