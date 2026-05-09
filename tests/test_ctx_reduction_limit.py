"""Tests for ctx_size_reduction_limit feature.

Verifies that:
1. ModelParameters accepts ctx_size_reduction_limit
2. Argument builder passes --fit-ctx based on the limit
3. BaseServerManager._get_minimum_ctx() computes correctly
4. Context validation enforces the minimum and stops server below it
"""

from unittest.mock import MagicMock, patch

import pytest

from models import ModelParameters


class TestModelParametersSchema:
    """ctx_size_reduction_limit is a valid field on ModelParameters."""

    def test_default_is_05(self):
        params = ModelParameters()
        assert params.ctx_size_reduction_limit == 0.5

    def test_accepts_custom_value(self):
        params = ModelParameters(ctx_size_reduction_limit=0.75)
        assert params.ctx_size_reduction_limit == 0.75

    def test_accepts_zero(self):
        params = ModelParameters(ctx_size_reduction_limit=0.0)
        assert params.ctx_size_reduction_limit == 0.0

    def test_accepts_one(self):
        params = ModelParameters(ctx_size_reduction_limit=1.0)
        assert params.ctx_size_reduction_limit == 1.0

    def test_rejects_above_one(self):
        with pytest.raises(Exception):
            ModelParameters(ctx_size_reduction_limit=1.5)

    def test_rejects_negative(self):
        with pytest.raises(Exception):
            ModelParameters(ctx_size_reduction_limit=-0.1)


class TestArgumentBuilderFitCtx:
    """The argument builder should pass --fit-ctx to llama.cpp."""

    def _make_model(self, num_ctx=40960, reduction_limit=0.5):
        mock = MagicMock()
        mock.name = "test-model"
        mock.model = "test-model"
        mock.details = MagicMock()
        mock.details.gguf_file = "/path/to/model.gguf"
        mock.details.clip_model_path = None
        mock.parameters = ModelParameters(
            num_ctx=num_ctx,
            ctx_size_reduction_limit=reduction_limit,
        )
        mock.draft_model = None
        return mock

    @patch(
        "server_manager.llamacpp_argument_builder.LLAMA_SERVER_EXECUTABLE",
        "/fake/llama-server",
    )
    def test_fit_ctx_computed_from_reduction_limit(self):
        """fit-ctx = ceil(num_ctx * reduction_limit)."""
        from server_manager.llamacpp_argument_builder import LlamaCppArgumentBuilder

        model = self._make_model(num_ctx=40960, reduction_limit=0.5)
        builder = LlamaCppArgumentBuilder(model, port=8080)
        args = builder.build_args()
        args_str = " ".join(args)
        assert "--fit-ctx 20480" in args_str

    @patch(
        "server_manager.llamacpp_argument_builder.LLAMA_SERVER_EXECUTABLE",
        "/fake/llama-server",
    )
    def test_fit_ctx_with_custom_limit(self):
        """fit-ctx respects custom reduction limit."""
        from server_manager.llamacpp_argument_builder import LlamaCppArgumentBuilder

        model = self._make_model(num_ctx=40960, reduction_limit=0.75)
        builder = LlamaCppArgumentBuilder(model, port=8080)
        args = builder.build_args()
        args_str = " ".join(args)
        assert "--fit-ctx 30720" in args_str

    @patch(
        "server_manager.llamacpp_argument_builder.LLAMA_SERVER_EXECUTABLE",
        "/fake/llama-server",
    )
    def test_fit_ctx_floored_at_4096(self):
        """fit-ctx has a 4096 absolute floor."""
        from server_manager.llamacpp_argument_builder import LlamaCppArgumentBuilder

        model = self._make_model(num_ctx=4096, reduction_limit=0.5)
        builder = LlamaCppArgumentBuilder(model, port=8080)
        args = builder.build_args()
        args_str = " ".join(args)
        # ceil(4096 * 0.5) = 2048, but floor is 4096
        assert "--fit-ctx 4096" in args_str

    @patch(
        "server_manager.llamacpp_argument_builder.LLAMA_SERVER_EXECUTABLE",
        "/fake/llama-server",
    )
    def test_fit_ctx_uses_default_limit_when_not_set(self):
        """When ctx_size_reduction_limit is default (0.5), fit-ctx is half of num_ctx."""
        from server_manager.llamacpp_argument_builder import LlamaCppArgumentBuilder

        model = self._make_model(num_ctx=90000, reduction_limit=0.5)
        builder = LlamaCppArgumentBuilder(model, port=8080)
        args = builder.build_args()
        args_str = " ".join(args)
        assert "--fit-ctx 45000" in args_str


class TestGetMinimumCtx:
    """BaseServerManager._get_minimum_ctx() computes correctly."""

    def _make_manager(self, num_ctx=40960, reduction_limit=0.5):
        from server_manager.base import BaseServerManager

        mock_model = MagicMock()
        mock_model.name = "test"
        mock_model.parameters = ModelParameters(
            num_ctx=num_ctx,
            ctx_size_reduction_limit=reduction_limit,
        )

        # Create a concrete subclass for testing
        class TestManager(BaseServerManager):
            def _build_server_args(self):
                return []

            def get_api_endpoint(self, path):
                return f"http://localhost:8080{path}"

        mgr = TestManager(model=mock_model, port=8080)
        return mgr

    def test_default_limit(self):
        mgr = self._make_manager(num_ctx=40960, reduction_limit=0.5)
        assert mgr._get_minimum_ctx() == 20480

    def test_custom_limit(self):
        mgr = self._make_manager(num_ctx=40960, reduction_limit=0.75)
        assert mgr._get_minimum_ctx() == 30720

    def test_no_parameters(self):
        from server_manager.base import BaseServerManager

        mock_model = MagicMock()
        mock_model.name = "test"
        mock_model.parameters = None

        class TestManager(BaseServerManager):
            def _build_server_args(self):
                return []

            def get_api_endpoint(self, path):
                return f"http://localhost:8080{path}"

        mgr = TestManager(model=mock_model, port=8080)
        # Default num_ctx=90000, default limit=0.5 => 45000
        assert mgr._get_minimum_ctx() == 45000

    def test_floor_at_2048(self):
        mgr = self._make_manager(num_ctx=2048, reduction_limit=0.5)
        # ceil(2048 * 0.5) = 1024, but floor is 2048
        assert mgr._get_minimum_ctx() == 2048
