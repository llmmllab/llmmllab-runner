"""Comprehensive tests for the priority request queue (task_queue.py).

Covers:
  - Priority enum ordering and values
  - RequestSource enum
  - QueuedRequest construction, ordering, aging
  - PriorityRequestQueue enqueue/dequeue/cancel/size/peek
  - Header parsing (parse_priority_header, parse_source_header)

Integration with harnesses
=========================
The priority queue is consumed by the server creation flow in
routers/servers.py (via app.py).  External clients signal priority
through HTTP headers:

  X-Request-Priority: high | medium | low
  X-Request-Source:   user | scheduled | system

Claude Code and OpenClaw agents typically send requests without these
headers, so they default to MEDIUM priority / USER source.  Scheduled
maintenance jobs (e.g., model warm-up) should set LOW priority.
"""

import asyncio
import importlib
import os
import sys
import time
from unittest.mock import patch

import pytest

# Ensure the project root is first on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import task_queue as queue_module  # noqa: E402

AGING_LOW_TO_MEDIUM = queue_module.AGING_LOW_TO_MEDIUM
AGING_MEDIUM_TO_HIGH = queue_module.AGING_MEDIUM_TO_HIGH
Priority = queue_module.Priority
PriorityRequestQueue = queue_module.PriorityRequestQueue
QueuedRequest = queue_module.QueuedRequest
RequestSource = queue_module.RequestSource
PRIORITY_MAP = queue_module.PRIORITY_MAP
SOURCE_MAP = queue_module.SOURCE_MAP
parse_priority_header = queue_module.parse_priority_header
parse_source_header = queue_module.parse_source_header


# ---------------------------------------------------------------------------
# Priority & RequestSource enum tests
# ---------------------------------------------------------------------------

class TestPriorityEnum:
    def test_priority_values(self):
        assert Priority.HIGH.value == 1
        assert Priority.MEDIUM.value == 2
        assert Priority.LOW.value == 3

    def test_priority_ordering(self):
        assert Priority.HIGH.value < Priority.MEDIUM.value < Priority.LOW.value


class TestRequestSourceEnum:
    def test_source_values(self):
        assert RequestSource.USER.value == "user"
        assert RequestSource.SCHEDULED.value == "scheduled"
        assert RequestSource.SYSTEM.value == "system"


# ---------------------------------------------------------------------------
# Header parsing tests
# ---------------------------------------------------------------------------

class TestParsePriorityHeader:
    def test_none_defaults_to_medium(self):
        assert parse_priority_header(None) is Priority.MEDIUM

    def test_valid_high(self):
        assert parse_priority_header("high") is Priority.HIGH

    def test_valid_medium(self):
        assert parse_priority_header("medium") is Priority.MEDIUM

    def test_valid_low(self):
        assert parse_priority_header("low") is Priority.LOW

    def test_case_insensitive(self):
        assert parse_priority_header("HIGH") is Priority.HIGH
        assert parse_priority_header("Low") is Priority.LOW
        assert parse_priority_header("MeDiUm") is Priority.MEDIUM

    def test_whitespace_stripped(self):
        assert parse_priority_header("  high  ") is Priority.HIGH

    def test_unknown_defaults_to_medium(self):
        assert parse_priority_header("urgent") is Priority.MEDIUM
        assert parse_priority_header("") is Priority.MEDIUM


class TestParseSourceHeader:
    def test_none_defaults_to_user(self):
        assert parse_source_header(None) is RequestSource.USER

    def test_valid_user(self):
        assert parse_source_header("user") is RequestSource.USER

    def test_valid_scheduled(self):
        assert parse_source_header("scheduled") is RequestSource.SCHEDULED

    def test_valid_system(self):
        assert parse_source_header("system") is RequestSource.SYSTEM

    def test_case_insensitive(self):
        assert parse_source_header("USER") is RequestSource.USER
        assert parse_source_header("Scheduled") is RequestSource.SCHEDULED

    def test_unknown_defaults_to_user(self):
        assert parse_source_header("unknown") is RequestSource.USER


# ---------------------------------------------------------------------------
# QueuedRequest tests
# ---------------------------------------------------------------------------

class TestQueuedRequest:
    def test_creation(self):
        req = QueuedRequest(priority=Priority.HIGH, request="test")
        assert req.priority is Priority.HIGH
        assert req.request == "test"
        assert req.source is RequestSource.USER
        assert req._original_priority is Priority.HIGH

    def test_sort_key_ordering(self):
        high = QueuedRequest(priority=Priority.HIGH, request="h", enqueue_time=100)
        low = QueuedRequest(priority=Priority.LOW, request="l", enqueue_time=100)
        assert high.sort_key < low.sort_key

    def test_fifo_within_same_priority(self):
        first = QueuedRequest(priority=Priority.MEDIUM, request="a", enqueue_time=100)
        second = QueuedRequest(priority=Priority.MEDIUM, request="b", enqueue_time=200)
        assert first.sort_key < second.sort_key

    def test_aging_low_to_medium(self):
        with patch("time.time", return_value=0):
            req = QueuedRequest(priority=Priority.LOW, request="r", enqueue_time=0)
        with patch("time.time", return_value=AGING_LOW_TO_MEDIUM):
            promoted = req.apply_aging()
        assert promoted is True
        assert req.priority is Priority.MEDIUM

    def test_aging_medium_to_high(self):
        with patch("time.time", return_value=0):
            req = QueuedRequest(priority=Priority.MEDIUM, request="r", enqueue_time=0)
        with patch("time.time", return_value=AGING_MEDIUM_TO_HIGH):
            promoted = req.apply_aging()
        assert promoted is True
        assert req.priority is Priority.HIGH

    def test_no_aging_when_not_elapsed(self):
        with patch("time.time", return_value=0):
            req = QueuedRequest(priority=Priority.LOW, request="r", enqueue_time=0)
        with patch("time.time", return_value=AGING_LOW_TO_MEDIUM - 1):
            promoted = req.apply_aging()
        assert promoted is False
        assert req.priority is Priority.LOW

    def test_high_priority_does_not_age(self):
        with patch("time.time", return_value=0):
            req = QueuedRequest(priority=Priority.HIGH, request="r", enqueue_time=0)
        with patch("time.time", return_value=9999):
            promoted = req.apply_aging()
        assert promoted is False
        assert req.priority is Priority.HIGH

    def test_aging_preserves_enqueue_time(self):
        original_time = 42.0
        with patch("time.time", return_value=0):
            req = QueuedRequest(priority=Priority.LOW, request="r", enqueue_time=original_time)
        with patch("time.time", return_value=AGING_LOW_TO_MEDIUM):
            req.apply_aging()
        assert req.enqueue_time == original_time


# ---------------------------------------------------------------------------
# PriorityRequestQueue tests
# ---------------------------------------------------------------------------

@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def make_queue():
    return PriorityRequestQueue()


class TestPriorityRequestQueueEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_returns_future(self):
        q = await make_queue()
        future = await q.enqueue("request-1")
        assert isinstance(future, asyncio.Future)
        assert not future.done()

    @pytest.mark.asyncio
    async def test_enqueue_increases_size(self):
        q = await make_queue()
        await q.enqueue("r1")
        await q.enqueue("r2")
        assert await q.size() == 2

    @pytest.mark.asyncio
    async def test_enqueue_with_priority(self):
        q = await make_queue()
        await q.enqueue("r1", priority=Priority.HIGH)
        await q.enqueue("r2", priority=Priority.LOW)
        assert await q.size() == 2

    @pytest.mark.asyncio
    async def test_enqueue_with_source(self):
        q = await make_queue()
        await q.enqueue("r1", source=RequestSource.SYSTEM)
        item = await q.dequeue()
        assert item.source is RequestSource.SYSTEM


class TestPriorityRequestQueueDequeue:
    @pytest.mark.asyncio
    async def test_dequeue_empty_returns_none(self):
        q = await make_queue()
        result = await q.dequeue()
        assert result is None

    @pytest.mark.asyncio
    async def test_dequeue_returns_highest_priority(self):
        q = await make_queue()
        await q.enqueue("low", priority=Priority.LOW)
        await q.enqueue("high", priority=Priority.HIGH)
        await q.enqueue("medium", priority=Priority.MEDIUM)

        item = await q.dequeue()
        assert item.request == "high"
        assert item.priority is Priority.HIGH

    @pytest.mark.asyncio
    async def test_dequeue_fifo_within_priority(self):
        q = await make_queue()
        await q.enqueue("first", priority=Priority.MEDIUM)
        await q.enqueue("second", priority=Priority.MEDIUM)
        await q.enqueue("third", priority=Priority.MEDIUM)

        item1 = await q.dequeue()
        item2 = await q.dequeue()
        item3 = await q.dequeue()

        assert item1.request == "first"
        assert item2.request == "second"
        assert item3.request == "third"

    @pytest.mark.asyncio
    async def test_dequeue_decreases_size(self):
        q = await make_queue()
        await q.enqueue("r1")
        assert await q.size() == 1
        await q.dequeue()
        assert await q.size() == 0

    @pytest.mark.asyncio
    async def test_dequeue_applies_aging(self):
        """A long-waiting LOW request should be promoted before dequeue."""
        q = await make_queue()
        # Enqueue a LOW request with an old timestamp
        old_future = asyncio.get_event_loop().create_future()
        old_item = QueuedRequest(
            priority=Priority.LOW,
            enqueue_time=time.time() - AGING_LOW_TO_MEDIUM - 1,
            request="old-low",
            source=RequestSource.USER,
            future=old_future,
        )
        # Manually insert into heap (bypassing enqueue to control timestamp)
        import heapq
        async with q._lock:
            heapq.heappush(q._heap, old_item)

        # Enqueue a fresh HIGH request
        await q.enqueue("fresh-high", priority=Priority.HIGH)

        # The HIGH should come out first
        item = await q.dequeue()
        assert item.request == "fresh-high"

        # The old LOW should have aged to MEDIUM
        item2 = await q.dequeue()
        assert item2.request == "old-low"
        assert item2.priority is Priority.MEDIUM


class TestPriorityRequestQueuePeek:
    @pytest.mark.asyncio
    async def test_peek_empty_returns_none(self):
        q = await make_queue()
        assert await q.peek() is None

    @pytest.mark.asyncio
    async def test_peek_returns_top_without_removing(self):
        q = await make_queue()
        await q.enqueue("r1", priority=Priority.LOW)
        await q.enqueue("r2", priority=Priority.HIGH)

        item = await q.peek()
        assert item.request == "r2"
        assert await q.size() == 2  # still there


class TestPriorityRequestQueueCancel:
    @pytest.mark.asyncio
    async def test_cancel_removes_request(self):
        q = await make_queue()
        future = await q.enqueue("r1")
        result = await q.cancel(future)
        assert result is True
        assert await q.size() == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self):
        q = await make_queue()
        loop = asyncio.get_event_loop()
        fake_future = loop.create_future()
        result = await q.cancel(fake_future)
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_cancels_future(self):
        q = await make_queue()
        future = await q.enqueue("r1")
        await q.cancel(future)
        assert future.done()
        assert future.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_preserves_other_items(self):
        q = await make_queue()
        f1 = await q.enqueue("r1")
        f2 = await q.enqueue("r2")
        f3 = await q.enqueue("r3")

        await q.cancel(f2)
        assert await q.size() == 2

        item = await q.dequeue()
        assert item.request in ("r1", "r3")


class TestPriorityRequestQueueSize:
    @pytest.mark.asyncio
    async def test_initial_size_is_zero(self):
        q = await make_queue()
        assert await q.size() == 0

    @pytest.mark.asyncio
    async def test_size_tracks_enqueue_and_dequeue(self):
        q = await make_queue()
        await q.enqueue("r1")
        await q.enqueue("r2")
        await q.enqueue("r3")
        assert await q.size() == 3

        await q.dequeue()
        assert await q.size() == 2

        await q.dequeue()
        await q.dequeue()
        assert await q.size() == 0


class TestPriorityRequestQueueApplyAging:
    @pytest.mark.asyncio
    async def test_apply_aging_to_all_promotes_old_items(self):
        q = await make_queue()
        # Insert an old LOW item
        old_future = asyncio.get_event_loop().create_future()
        old_item = QueuedRequest(
            priority=Priority.LOW,
            enqueue_time=time.time() - AGING_LOW_TO_MEDIUM - 1,
            request="old",
            future=old_future,
        )
        import heapq
        async with q._lock:
            heapq.heappush(q._heap, old_item)

        await q.apply_aging_to_all()

        # Verify the item was promoted
        item = await q.peek()
        assert item.request == "old"
        assert item.priority is Priority.MEDIUM


class TestPriorityRequestQueueIntegration:
    @pytest.mark.asyncio
    async def test_full_lifecycle_enqueue_dequeue_cancel(self):
        """End-to-end: enqueue multiple, cancel one, dequeue rest in order."""
        q = await make_queue()

        f_low = await q.enqueue("low-task", priority=Priority.LOW)
        f_high = await q.enqueue("high-task", priority=Priority.HIGH)
        f_med = await q.enqueue("med-task", priority=Priority.MEDIUM)

        # Cancel the medium task
        await q.cancel(f_med)
        assert await q.size() == 2

        # Dequeue should return HIGH first, then LOW
        item1 = await q.dequeue()
        assert item1.request == "high-task"

        item2 = await q.dequeue()
        assert item2.request == "low-task"

        # Queue should be empty
        assert await q.size() == 0
        assert await q.dequeue() is None

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_dequeue(self):
        """Multiple enqueues and dequeues should not corrupt the queue."""
        q = await make_queue()

        futures = []
        for i in range(10):
            pri = Priority.HIGH if i % 3 == 0 else (Priority.MEDIUM if i % 3 == 1 else Priority.LOW)
            f = await q.enqueue(f"task-{i}", priority=pri)
            futures.append(f)

        assert await q.size() == 10

        results = []
        for _ in range(10):
            item = await q.dequeue()
            results.append(item.request)

        assert len(results) == 10
        assert await q.size() == 0

    @pytest.mark.asyncio
    async def test_priority_ordering_across_all_levels(self):
        """Verify strict priority ordering: HIGH > MEDIUM > LOW."""
        q = await make_queue()

        # Enqueue in reverse priority order
        await q.enqueue("low-1", priority=Priority.LOW)
        await q.enqueue("low-2", priority=Priority.LOW)
        await q.enqueue("med-1", priority=Priority.MEDIUM)
        await q.enqueue("high-1", priority=Priority.HIGH)
        await q.enqueue("med-2", priority=Priority.MEDIUM)
        await q.enqueue("high-2", priority=Priority.HIGH)

        results = []
        while await q.size() > 0:
            item = await q.dequeue()
            results.append(item.request)

        # HIGH items first (FIFO within), then MEDIUM, then LOW
        assert results[0] == "high-1"
        assert results[1] == "high-2"
        assert results[2] == "med-1"
        assert results[3] == "med-2"
        assert results[4] == "low-1"
        assert results[5] == "low-2"
