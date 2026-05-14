"""Tests for persistent KV cache slot save/restore feature (issue #33)."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestSlotPersistenceConfig:
    """Test that slot persistence config variables are read correctly."""

    def test_slot_save_dir_default_empty(self):
        """SLOT_SAVE_DIR defaults to empty string (disabled)."""
        # Remove any env override
        env = os.environ.copy()
        env.pop("SLOT_SAVE_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            # Force reimport
            import importlib

            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SLOT_SAVE_DIR == ""

    def test_slot_no_mmap_default_true(self):
        """SLOT_NO_MMAP defaults to True."""
        env = os.environ.copy()
        env.pop("SLOT_NO_MMAP", None)
        with patch.dict(os.environ, env, clear=True):
            import importlib

            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SLOT_NO_MMAP is True

    def test_slot_swa_full_default_true(self):
        """SLOT_SWA_FULL defaults to True."""
        env = os.environ.copy()
        env.pop("SLOT_SWA_FULL", None)
        with patch.dict(os.environ, env, clear=True):
            import importlib

            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SLOT_SWA_FULL is True


class TestSlotPersistenceArgumentBuilder:
    """Test that LlamaCppArgumentBuilder includes slot flags when configured."""

    def _make_model(self, **kwargs):
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

    def test_no_slot_flags_when_disabled(self):
        """No slot-related flags when SLOT_SAVE_DIR is empty."""
        with patch.dict(os.environ, {"SLOT_SAVE_DIR": ""}):
            # Force fresh import
            import importlib

            if "config" in sys.modules:
                del sys.modules["config"]
            if "server_manager.llamacpp_argument_builder" in sys.modules:
                del sys.modules["server_manager.llamacpp_argument_builder"]

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from server_manager.llamacpp_argument_builder import (
                LlamaCppArgumentBuilder,
            )

            model = self._make_model()
            builder = LlamaCppArgumentBuilder(model=model, port=9999)
            args = builder.build_args()
            args_str = " ".join(args)

            assert "--slot-save-path" not in args_str
            assert "--no-mmap" not in args_str
            assert "--swa-full" not in args_str

    def test_slot_flags_when_enabled(self):
        """Slot flags appear when SLOT_SAVE_DIR is set."""
        with patch.dict(
            os.environ,
            {
                "SLOT_SAVE_DIR": "/data/slots",
                "SLOT_NO_MMAP": "true",
                "SLOT_SWA_FULL": "true",
            },
        ):
            import importlib

            for mod in list(sys.modules):
                if mod.startswith("config") or mod.startswith("server_manager"):
                    del sys.modules[mod]

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from server_manager.llamacpp_argument_builder import (
                LlamaCppArgumentBuilder,
            )

            model = self._make_model()
            builder = LlamaCppArgumentBuilder(model=model, port=9999)
            args = builder.build_args()
            args_str = " ".join(args)

            assert "--slot-save-path" in args_str
            assert "/data/slots" in args_str
            assert "--no-mmap" in args_str
            assert "--swa-full" in args_str

    def test_no_mmap_can_be_disabled(self):
        """--no-mmap is omitted when SLOT_NO_MMAP=false."""
        with patch.dict(
            os.environ,
            {
                "SLOT_SAVE_DIR": "/data/slots",
                "SLOT_NO_MMAP": "false",
                "SLOT_SWA_FULL": "true",
            },
        ):
            import importlib

            for mod in list(sys.modules):
                if mod.startswith("config") or mod.startswith("server_manager"):
                    del sys.modules[mod]

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from server_manager.llamacpp_argument_builder import (
                LlamaCppArgumentBuilder,
            )

            model = self._make_model()
            builder = LlamaCppArgumentBuilder(model=model, port=9999)
            args = builder.build_args()
            args_str = " ".join(args)

            assert "--slot-save-path" in args_str
            assert "--no-mmap" not in args_str
            assert "--swa-full" in args_str

    def test_swa_full_can_be_disabled(self):
        """--swa-full is omitted when SLOT_SWA_FULL=false."""
        with patch.dict(
            os.environ,
            {
                "SLOT_SAVE_DIR": "/data/slots",
                "SLOT_NO_MMAP": "true",
                "SLOT_SWA_FULL": "false",
            },
        ):
            import importlib

            for mod in list(sys.modules):
                if mod.startswith("config") or mod.startswith("server_manager"):
                    del sys.modules[mod]

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from server_manager.llamacpp_argument_builder import (
                LlamaCppArgumentBuilder,
            )

            model = self._make_model()
            builder = LlamaCppArgumentBuilder(model=model, port=9999)
            args = builder.build_args()
            args_str = " ".join(args)

            assert "--slot-save-path" in args_str
            assert "--no-mmap" in args_str
            assert "--swa-full" not in args_str


import sys
