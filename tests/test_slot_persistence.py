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
