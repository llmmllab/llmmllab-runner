"""Tests for persistent KV cache slot save/restore feature (issue #33)."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _clear_slot_modules():
    """Remove cached config/server_manager modules so env changes take effect."""
    for mod in list(sys.modules):
        if mod.startswith("config") or mod.startswith("server_manager"):
            del sys.modules[mod]


def _import_builder():
    """Fresh-import LlamaCppArgumentBuilder after module cache is cleared."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from server_manager.llamacpp_argument_builder import (
        LlamaCppArgumentBuilder,
    )
    return LlamaCppArgumentBuilder


def _make_model():
    """Create a minimal mock model for testing."""
    model = MagicMock()
    model.id = "test-model"
    model.name = "test-model"
    model.model = "/path/to/model.gguf"
    model.details = MagicMock()
    model.details.gguf_file = "/path/to/model.gguf"
    model.details.clip_model_path = None
    model.details.size = 0
    model.parameters = MagicMock()
    model.parameters.num_ctx = 8192
    model.parameters.batch_size = 2048
    model.parameters.micro_batch_size = None
    model.parameters.repeat_penalty = 1.1
    model.parameters.repeat_last_n = None
    model.parameters.n_gpu_layers = None
    model.parameters.main_gpu = None
    model.parameters.split_mode = "layer"
    model.parameters.tensor_split = None
    model.parameters.kv_on_cpu = False
    model.parameters.parallel = 4
    model.parameters.think = False
    model.parameters.ctx_size_reduction_limit = 0.5
    model.draft_model = None
    return model


class TestSlotPersistenceConfig:
    """Test that slot persistence config variables are read correctly."""

    def test_slot_save_dir_default_empty(self):
        """SLOT_SAVE_DIR defaults to empty string (disabled)."""
        env = os.environ.copy()
        env.pop("SLOT_SAVE_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            _clear_slot_modules()
            import config

            assert config.SLOT_SAVE_DIR == ""

    def test_slot_no_mmap_default_false(self):
        """SLOT_NO_MMAP defaults to False.

        Was True historically (forced on whenever SLOT_SAVE_DIR was set);
        flipped in commit cdb05f0 after we confirmed --no-mmap is not
        actually required for slot persistence — only a platform-specific
        workaround per the upstream tutorial.  Forcing it on caused
        OOMKilled crash loops because llama.cpp loaded entire model
        files into host RAM.
        """
        env = os.environ.copy()
        env.pop("SLOT_NO_MMAP", None)
        with patch.dict(os.environ, env, clear=True):
            _clear_slot_modules()
            import config

            assert config.SLOT_NO_MMAP is False

    def test_slot_swa_full_default_true(self):
        """SLOT_SWA_FULL defaults to True."""
        env = os.environ.copy()
        env.pop("SLOT_SWA_FULL", None)
        with patch.dict(os.environ, env, clear=True):
            _clear_slot_modules()
            import config

            assert config.SLOT_SWA_FULL is True


class TestSlotPersistenceArgumentBuilder:
    """Test that LlamaCppArgumentBuilder includes slot flags when configured."""

    def test_no_slot_flags_when_disabled(self):
        """No slot-related flags when SLOT_SAVE_DIR is empty."""
        env = os.environ.copy()
        env.pop("SLOT_SAVE_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            _clear_slot_modules()
            Builder = _import_builder()

            builder = Builder(model=_make_model(), port=9999)
            args_str = " ".join(builder.build_args())

            assert "--slot-save-path" not in args_str
            assert "--no-mmap" not in args_str
            assert "--swa-full" not in args_str

    def test_slot_flags_when_enabled(self):
        """Slot flags appear when SLOT_SAVE_DIR is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            slot_dir = os.path.join(tmpdir, "slots")
            with patch.dict(
                os.environ,
                {
                    "SLOT_SAVE_DIR": slot_dir,
                    "SLOT_NO_MMAP": "true",
                    "SLOT_SWA_FULL": "true",
                },
            ):
                _clear_slot_modules()
                Builder = _import_builder()

                builder = Builder(model=_make_model(), port=9999)
                args_str = " ".join(builder.build_args())

                assert "--slot-save-path" in args_str
                assert slot_dir in args_str
                assert "--no-mmap" in args_str
                assert "--swa-full" in args_str

    def test_no_mmap_can_be_disabled(self):
        """--no-mmap is omitted when SLOT_NO_MMAP=false."""
        with tempfile.TemporaryDirectory() as tmpdir:
            slot_dir = os.path.join(tmpdir, "slots")
            with patch.dict(
                os.environ,
                {
                    "SLOT_SAVE_DIR": slot_dir,
                    "SLOT_NO_MMAP": "false",
                    "SLOT_SWA_FULL": "true",
                },
            ):
                _clear_slot_modules()
                Builder = _import_builder()

                builder = Builder(model=_make_model(), port=9999)
                args_str = " ".join(builder.build_args())

                assert "--slot-save-path" in args_str
                assert "--no-mmap" not in args_str
                assert "--swa-full" in args_str

    def test_swa_full_can_be_disabled(self):
        """--swa-full is omitted when SLOT_SWA_FULL=false."""
        with tempfile.TemporaryDirectory() as tmpdir:
            slot_dir = os.path.join(tmpdir, "slots")
            with patch.dict(
                os.environ,
                {
                    "SLOT_SAVE_DIR": slot_dir,
                    "SLOT_NO_MMAP": "true",
                    "SLOT_SWA_FULL": "false",
                },
            ):
                _clear_slot_modules()
                Builder = _import_builder()

                builder = Builder(model=_make_model(), port=9999)
                args_str = " ".join(builder.build_args())

                assert "--slot-save-path" in args_str
                assert "--no-mmap" in args_str
                assert "--swa-full" not in args_str

    def test_slot_save_dir_created_when_missing(self):
        """SLOT_SAVE_DIR directory is auto-created by the argument builder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "deep" / "nested" / "slots"
            assert not new_dir.exists()

            with patch.dict(
                os.environ,
                {
                    "SLOT_SAVE_DIR": str(new_dir),
                    "SLOT_NO_MMAP": "true",
                    "SLOT_SWA_FULL": "true",
                },
            ):
                _clear_slot_modules()
                Builder = _import_builder()

                builder = Builder(model=_make_model(), port=9999)
                builder.build_args()

                assert new_dir.exists()
                assert new_dir.is_dir()

    def test_slot_save_dir_missing_logs_warning(self):
        """If SLOT_SAVE_DIR cannot be created, a warning is logged but no crash."""
        # Use a path that cannot be created (read-only /nonexistent)
        with patch.dict(
            os.environ,
            {
                "SLOT_SAVE_DIR": "/nonexistent/deep/slots",
                "SLOT_NO_MMAP": "true",
                "SLOT_SWA_FULL": "true",
            },
        ):
            _clear_slot_modules()
            Builder = _import_builder()

            builder = Builder(model=_make_model(), port=9999)
            # Should not raise — mkdir failure is caught and logged
            args_str = " ".join(builder.build_args())

            # Flags should still be present even though dir creation failed
            assert "--slot-save-path" in args_str
            assert "/nonexistent/deep/slots" in args_str


class TestSlotProxyHelpers:
    """Test slot file path generation and slot ID resolution in proxy router."""

    def test_slot_file_path(self):
        """Slot file path is built correctly from session_id."""
        import importlib

        env = os.environ.copy()
        env["SLOT_SAVE_DIR"] = "/data/slots"
        with patch.dict(os.environ, env, clear=True):
            for mod in list(sys.modules):
                if mod.startswith("config") or mod.startswith("proxy"):
                    del sys.modules[mod]
            import config
            importlib.reload(config)
            from proxy.router import _slot_file_path

            path = _slot_file_path("session-abc-123")
            assert path == "/data/slots/slot_session-abc-123.bin"

            # With server_id, file is model-specific to prevent cross-model corruption
            path_with_server = _slot_file_path("session-abc-123", "srv-xyz")
            assert path_with_server == "/data/slots/slot_session-abc-123_srv-xyz.bin"

    def test_resolve_slot_id_consistent(self):
        """Same session_id always maps to the same slot."""
        from proxy.router import _resolve_slot_id

        session = "my-session-id"
        slot_a = _resolve_slot_id(session, "server-1", 4)
        slot_b = _resolve_slot_id(session, "server-1", 4)
        assert slot_a == slot_b

    def test_resolve_slot_id_different_sessions(self):
        """Different session_ids may map to different slots."""
        from proxy.router import _resolve_slot_id

        # With enough sessions, at least two should land on different slots
        slots_seen = set()
        for i in range(10):
            sid = _resolve_slot_id(f"session-{i}", "server-1", 4)
            slots_seen.add(sid)
        # With 10 sessions and 4 slots, we should see multiple slots used
        assert len(slots_seen) > 1

    def test_resolve_slot_id_range(self):
        """Slot ID is always within valid range."""
        from proxy.router import _resolve_slot_id

        for num_slots in [1, 2, 4, 8, 16, 64, 256]:
            for i in range(20):
                sid = _resolve_slot_id(f"session-{i}", "server-1", num_slots)
                assert 0 <= sid < num_slots, (
                    f"slot {sid} out of range for {num_slots} slots"
                )

    def test_resolve_slot_id_single_slot(self):
        """With 1 slot, all sessions map to slot 0."""
        from proxy.router import _resolve_slot_id

        for i in range(10):
            sid = _resolve_slot_id(f"session-{i}", "server-1", 1)
            assert sid == 0


class TestSlotLRU:
    """Tests for the per-server SlotLRU added in Round 1.

    Replaces the older hash-then-capture pattern (_session_slot_cache,
    x-slot-id header). The runner now eagerly assigns each session a slot
    via a fixed-capacity LRU keyed by session_id and injects ``id_slot``
    into the upstream JSON body before forwarding. llama.cpp respects
    explicit id_slot, so the assignment is authoritative — no "actual vs
    hash" mismatch to capture from a response header.
    """

    def test_first_session_gets_lowest_free_slot(self):
        """A brand-new session is assigned slot 0 in a fresh LRU."""
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=4)
        slot_id, evicted = asyncio.run(lru.touch("session-a"))
        assert slot_id == 0
        assert evicted is None

    def test_same_session_returns_same_slot(self):
        """touch() on an already-pinned session returns the same slot, no eviction."""
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=4)

        async def _run():
            s1, e1 = await lru.touch("sticky")
            s2, e2 = await lru.touch("sticky")
            s3, e3 = await lru.touch("sticky")
            return s1, s2, s3, (e1, e2, e3)

        s1, s2, s3, evicts = asyncio.run(_run())
        assert s1 == s2 == s3
        assert all(e is None for e in evicts)

    def test_distinct_sessions_get_distinct_slots(self):
        """Up to capacity, each new session_id claims a different slot."""
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=4)

        async def _run():
            results = []
            for sid in ("a", "b", "c", "d"):
                slot_id, evicted = await lru.touch(sid)
                results.append((slot_id, evicted))
            return results

        results = asyncio.run(_run())
        slot_ids = [r[0] for r in results]
        evictions = [r[1] for r in results]
        assert sorted(slot_ids) == [0, 1, 2, 3]
        assert all(e is None for e in evictions)

    def test_capacity_exceeded_evicts_lru_and_reuses_slot(self):
        """When the LRU is full, the next new session evicts the oldest and reuses its slot."""
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=2)

        async def _run():
            # Fill capacity. 'a' is LRU, 'b' is MRU.
            s_a, _ = await lru.touch("a")
            s_b, _ = await lru.touch("b")
            # A new session 'c' should evict 'a' (oldest) and reuse its slot.
            s_c, evicted = await lru.touch("c")
            return s_a, s_b, s_c, evicted

        s_a, s_b, s_c, evicted = asyncio.run(_run())
        assert {s_a, s_b} == {0, 1}
        assert evicted == ("a", s_a)
        assert s_c == s_a  # the evicted slot is reused

    def test_touch_refreshes_recency(self):
        """touch() bumps a session to MRU so it survives an eviction round."""
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=2)

        async def _run():
            await lru.touch("a")
            await lru.touch("b")
            # Re-touching 'a' should make 'b' the LRU.
            await lru.touch("a")
            # Inserting 'c' should now evict 'b', not 'a'.
            _, evicted = await lru.touch("c")
            return evicted

        evicted = asyncio.run(_run())
        assert evicted is not None
        assert evicted[0] == "b"

    def test_peek_does_not_touch_recency(self):
        """peek() returns the slot for a session without making it MRU."""
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=2)

        async def _run():
            slot_a, _ = await lru.touch("a")
            await lru.touch("b")  # 'b' is now MRU; 'a' is LRU
            # peek shouldn't change recency
            peeked = await lru.peek("a")
            _, evicted = await lru.touch("c")  # should still evict 'a'
            return slot_a, peeked, evicted

        slot_a, peeked, evicted = asyncio.run(_run())
        assert peeked == slot_a
        assert evicted is not None and evicted[0] == "a"

    def test_peek_returns_none_for_unknown_session(self):
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=4)
        assert asyncio.run(lru.peek("ghost")) is None

    def test_slot_ids_always_in_range(self):
        """Slot ids returned by touch() must always be in [0, capacity)."""
        import asyncio
        from proxy.router import SlotLRU

        lru = SlotLRU(capacity=3)

        async def _run():
            ids = []
            for sid in (f"s{i}" for i in range(10)):  # 10 sessions, capacity 3
                slot_id, _ = await lru.touch(sid)
                ids.append(slot_id)
            return ids

        ids = asyncio.run(_run())
        assert all(0 <= s < 3 for s in ids)
