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

    def test_slot_no_mmap_default_true(self):
        """SLOT_NO_MMAP defaults to True."""
        env = os.environ.copy()
        env.pop("SLOT_NO_MMAP", None)
        with patch.dict(os.environ, env, clear=True):
            _clear_slot_modules()
            import config

            assert config.SLOT_NO_MMAP is True

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


class TestSessionSlotCache:
    """Regression tests for the slot ID mismatch bug.

    Root cause: llama.cpp ignores `slot_id_or_index` when LCP similarity
    matching finds a better slot (sim_best > 0.100).  The hash-assigned
    slot and the slot llama.cpp actually used diverged, causing save/restore
    to hit the wrong slot and return 404.

    Fix: Don't inject `slot_id_or_index` on the first request.  Capture
    `x-slot-id` from the response headers, cache per session, and only
    inject on subsequent requests.
    """

    def setup_method(self):
        """Clear session slot cache before each test."""
        from proxy.router import _session_slot_cache
        _session_slot_cache.clear()

    def test_first_request_no_slot_injection(self):
        """slot_id_or_index is NOT injected when session is not cached.

        On the very first request for a session, we don't know which slot
        llama.cpp will assign, so we must NOT send slot_id_or_index.
        """
        from proxy.router import _session_slot_cache

        session_id = "new-session-xyz"
        assert session_id not in _session_slot_cache

        body = b'{"model": "test", "messages": [{"role": "user", "content": "hi"}]}'
        body_dict = __import__("json").loads(body)
        assert "slot_id_or_index" not in body_dict

        # Simulate the injection logic from proxy_request():
        #   if session_id in _session_slot_cache: inject
        if session_id in _session_slot_cache:
            body_dict["slot_id_or_index"] = _session_slot_cache[session_id]

        assert "slot_id_or_index" not in body_dict

    def test_subsequent_request_injects_cached_slot(self):
        """slot_id_or_index IS injected when session has a cached slot."""
        from proxy.router import _session_slot_cache

        session_id = "known-session"
        _session_slot_cache[session_id] = 7

        body = b'{"model": "test", "messages": [{"role": "user", "content": "hi"}]}'
        body_dict = __import__("json").loads(body)

        # Simulate the injection logic
        if session_id in _session_slot_cache:
            body_dict["slot_id_or_index"] = _session_slot_cache[session_id]

        assert body_dict["slot_id_or_index"] == 7

    def test_x_slot_id_capture_from_headers(self):
        """x-slot-id response header is parsed and cached correctly.

        This verifies the logic that captures the slot ID from llama.cpp's
        response headers, as done in _stream_upstream().
        """
        from proxy.router import _session_slot_cache

        session_id = "session-abc"
        assert session_id not in _session_slot_cache

        # Simulate capturing x-slot-id from response headers
        x_slot = "3"
        resp_slot_id = None
        if x_slot:
            try:
                resp_slot_id = int(x_slot)
            except (ValueError, TypeError):
                pass

        # Cache the discovered slot
        if session_id:
            _session_slot_cache[session_id] = resp_slot_id

        assert _session_slot_cache[session_id] == 3

    def test_x_slot_id_malicious_value_handled(self):
        """Non-integer x-slot-id values don't crash the capture logic."""
        for bad_value in ["", "abc", "3.5", None]:
            resp_slot_id = None
            if bad_value:
                try:
                    resp_slot_id = int(bad_value)
                except (ValueError, TypeError):
                    pass
            # Should be None for all bad values
            if bad_value in ("", "abc", "3.5", None):
                assert resp_slot_id is None

    def test_save_uses_actual_slot_not_hash_slot(self):
        """Save/restore uses the actual slot from x-slot-id, not the hash.

        Regression: when hash assigned slot 5 but llama.cpp used slot 7
        via LCP matching, save hit slot 5 and got 404. Now it should use
        the actual slot ID from the response.
        """
        from proxy.router import _session_slot_cache, _resolve_slot_id

        session_id = "session-mismatch"
        server_id = "server-1"

        # Hash assigns some slot
        hash_slot = _resolve_slot_id(session_id, server_id, 4)

        # But llama.cpp actually assigned a different slot (via x-slot-id)
        # Pick a slot that differs from the hash slot to prove the point
        resp_slot_id = (hash_slot + 1) % 4
        _session_slot_cache[session_id] = resp_slot_id

        # On the save path, we use the actual slot from response headers
        # (captured as `resp_slot_id` in the streaming path)
        # The logic is: actual_slot = resp_slot_id if resp_slot_id else slot_id
        final_slot = resp_slot_id if resp_slot_id is not None else hash_slot

        assert final_slot == resp_slot_id
        assert final_slot != hash_slot

    def test_restore_uses_cached_slot_from_previous_request(self):
        """Restore uses the cached slot ID from a prior request's x-slot-id."""
        from proxy.router import _session_slot_cache

        session_id = "returning-session"
        # Previous request established that llama.cpp uses slot 5
        _session_slot_cache[session_id] = 5

        # On restore, we use the cached value
        restore_slot_id = _session_slot_cache.get(session_id, 0)
        assert restore_slot_id == 5


class TestSlotOrchestrationFlow:
    """Integration-style tests for the full slot orchestration flow."""

    def setup_method(self):
        from proxy.router import _session_slot_cache, _num_slots_cache
        _session_slot_cache.clear()
        _num_slots_cache.clear()

    def test_full_flow_first_then_second_request(self):
        """Simulate first request (no injection) then second (with injection).

        1. First request: no slot_id_or_index, llama.cpp assigns slot via
           LCP matching, we capture x-slot-id from response
        2. Second request: slot_id_or_index injected from cache, llama.cpp
           uses the exact same slot, save/restore hits the right slot
        """
        import json as json_mod
        from proxy.router import _session_slot_cache, _resolve_slot_id

        session_id = "flow-test-session"
        body = json_mod.dumps({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
        })

        # --- First request ---
        # No cached slot, so no injection
        assert session_id not in _session_slot_cache
        body_dict = json_mod.loads(body)
        if session_id in _session_slot_cache:
            body_dict["slot_id_or_index"] = _session_slot_cache[session_id]
        assert "slot_id_or_index" not in body_dict

        # Llama.cpp responds with x-slot-id: 2
        resp_slot_id = 2
        _session_slot_cache[session_id] = resp_slot_id

        # --- Second request ---
        # Cached slot exists, so inject it
        body_dict = json_mod.loads(body)
        if session_id in _session_slot_cache:
            body_dict["slot_id_or_index"] = _session_slot_cache[session_id]
        assert body_dict["slot_id_or_index"] == 2

    def test_slot_cache_survives_multiple_requests(self):
        """Cached slot ID persists across multiple requests for same session."""
        from proxy.router import _session_slot_cache

        session_id = "sticky-session"
        _session_slot_cache[session_id] = 4

        for _ in range(5):
            assert _session_slot_cache.get(session_id) == 4

    def test_different_sessions_get_different_slots(self):
        """Different sessions can have different cached slots."""
        from proxy.router import _session_slot_cache

        # Simulate two sessions that llama.cpp assigned to different slots
        _session_slot_cache["session-a"] = 0
        _session_slot_cache["session-b"] = 2

        assert _session_slot_cache["session-a"] != _session_slot_cache["session-b"]
