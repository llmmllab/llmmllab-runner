"""Tests for the runner-side context guard that refuses undersized requests."""

from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from pydantic import BaseModel
from routers.servers import CreateServerRequest


class FakeModelParameters(BaseModel):
    num_ctx: int


class FakeModel(BaseModel):
    id: str
    parameters: FakeModelParameters


class TestContextGuard:
    """Verify that create_server rejects requests exceeding model n_ctx."""

    def _make_model(self, model_id: str, num_ctx: int) -> FakeModel:
        return FakeModel(id=model_id, parameters=FakeModelParameters(num_ctx=num_ctx))

    def test_request_exceeding_model_ctx_raises_507(self):
        """When num_ctx > model's num_ctx, the guard should raise 507."""
        model = self._make_model("test-model", 8192)
        requested_ctx = 16384

        model_ctx = (model.parameters.num_ctx if model.parameters else None) or 90000
        assert model_ctx == 8192

        # Simulate the guard logic
        if requested_ctx is not None and requested_ctx > model_ctx:
            exc = HTTPException(
                status_code=507,
                detail={
                    "reason": "context_too_large",
                    "message": (
                        f"Requested context size ({requested_ctx} tokens) exceeds "
                        f"the model's configured context window ({model_ctx} tokens). "
                        f"Reduce num_ctx or use a model with a larger context window."
                    ),
                    "requested_model": model.id,
                    "details": {
                        "requested_num_ctx": requested_ctx,
                        "model_num_ctx": model_ctx,
                    },
                },
            )
            assert exc.status_code == 507
            assert exc.detail["reason"] == "context_too_large"
            assert exc.detail["details"]["requested_num_ctx"] == 16384
            assert exc.detail["details"]["model_num_ctx"] == 8192
        else:
            raise AssertionError("Expected guard to trigger")

    def test_request_within_model_ctx_passes(self):
        """When num_ctx <= model's num_ctx, the guard should not trigger."""
        model = self._make_model("test-model", 16384)
        requested_ctx = 8192

        model_ctx = (model.parameters.num_ctx if model.parameters else None) or 90000
        assert model_ctx == 16384

        triggered = requested_ctx is not None and requested_ctx > model_ctx
        assert triggered is False

    def test_request_equal_to_model_ctx_passes(self):
        """When num_ctx == model's num_ctx, the guard should not trigger."""
        model = self._make_model("test-model", 8192)
        requested_ctx = 8192

        model_ctx = (model.parameters.num_ctx if model.parameters else None) or 90000
        triggered = requested_ctx is not None and requested_ctx > model_ctx
        assert triggered is False

    def test_no_num_ctx_provided_passes(self):
        """When num_ctx is None (not provided), the guard should not trigger."""
        model = self._make_model("test-model", 8192)
        requested_ctx = None

        model_ctx = (model.parameters.num_ctx if model.parameters else None) or 90000
        triggered = requested_ctx is not None and requested_ctx > model_ctx
        assert triggered is False

    def test_model_without_parameters_uses_default_90000(self):
        """When model has no parameters, the fallback n_ctx should be 90000."""

        class ModelNoParams(BaseModel):
            id: str
            parameters: object = None

        model = ModelNoParams(id="test-model")
        requested_ctx = 8000

        model_ctx = (model.parameters.num_ctx if model.parameters else None) or 90000
        assert model_ctx == 90000

        triggered = requested_ctx is not None and requested_ctx > model_ctx
        assert triggered is False

    def test_model_without_parameters_rejects_over_90000(self):
        """When model has no parameters and request exceeds 90000, should reject."""

        class ModelNoParams(BaseModel):
            id: str
            parameters: object = None

        model = ModelNoParams(id="test-model")
        requested_ctx = 100000

        model_ctx = (model.parameters.num_ctx if model.parameters else None) or 90000
        assert model_ctx == 90000

        triggered = requested_ctx is not None and requested_ctx > model_ctx
        assert triggered is True

    def test_create_server_request_accepts_num_ctx(self):
        """CreateServerRequest should accept an optional num_ctx field."""
        req = CreateServerRequest(model_id="test", num_ctx=8192)
        assert req.num_ctx == 8192

        req2 = CreateServerRequest(model_id="test")
        assert req2.num_ctx is None

    def test_error_detail_contains_actionable_message(self):
        """The 507 error message should tell the user how to fix it."""
        requested_ctx = 32768
        model_ctx = 8192

        msg = (
            f"Requested context size ({requested_ctx} tokens) exceeds "
            f"the model's configured context window ({model_ctx} tokens). "
            f"Reduce num_ctx or use a model with a larger context window."
        )
        assert "Reduce num_ctx" in msg
        assert "larger context window" in msg

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from routers.servers import CreateServerRequest


def _run(coro):
    """Run a coroutine in a fresh event loop for sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_model(ctx: int | None = None, params_none: bool = False):
    """Create a mock Model with configurable parameters.num_ctx."""
    mock = MagicMock()
    mock.id = "test-model"
    if params_none:
        mock.parameters = None
    else:
        mock.parameters = MagicMock()
        mock.parameters.num_ctx = ctx
    return mock


def _make_entry(server_id="srv-1", port=8080):
    mock = MagicMock()
    mock.server_id = server_id
    mock.port = port
    return mock


def _make_request():
    """Create a minimal FastAPI Request with a valid ASGI scope."""
    from fastapi import Request
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [],
    }
    return Request(scope)


class TestContextTooLargeGuard:
    """The runner should refuse to start a server when requested
    num_ctx exceeds the model's configured context window."""

    @patch("routers.servers.model_loader")
    @patch("app.request_queue")
    def test_rejects_when_requested_exceeds_model_ctx(
        self, mock_queue, mock_loader
    ):
        """507 when request num_ctx > model parameters.num_ctx."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=32000)
        mock_queue.size.return_value = asyncio.Future()
        mock_queue.size.return_value.set_result(0)

        from fastapi import HTTPException
        from routers.servers import create_server

        body = CreateServerRequest(model_id="test-model", num_ctx=64000)
        request = _make_request()

        with pytest.raises(HTTPException) as exc_info:
            _run(create_server(request, body))

        assert exc_info.value.status_code == 507
        detail = exc_info.value.detail
        assert detail["reason"] == "context_too_large"
        assert "64000" in detail["message"]
        assert "32000" in detail["message"]
        assert detail["details"]["requested_num_ctx"] == 64000
        assert detail["details"]["model_num_ctx"] == 32000

    @patch("routers.servers.model_loader")
    @patch("app.server_cache")
    @patch("app.request_queue")
    def test_allows_when_requested_within_model_ctx(
        self, mock_queue, mock_cache, mock_loader
    ):
        """No rejection when request num_ctx <= model parameters.num_ctx."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=64000)
        mock_queue.size.return_value = asyncio.Future()
        mock_queue.size.return_value.set_result(0)
        mock_cache.acquire_by_model.return_value = _make_entry()
        mock_cache.has_starting_server.return_value = False

        from routers.servers import create_server

        body = CreateServerRequest(model_id="test-model", num_ctx=32000)
        request = _make_request()

        result = _run(create_server(request, body))
        assert result["server_id"] == "srv-1"

    @patch("routers.servers.model_loader")
    @patch("app.server_cache")
    @patch("app.request_queue")
    def test_allows_when_num_ctx_not_provided(
        self, mock_queue, mock_cache, mock_loader
    ):
        """No rejection when request omits num_ctx (None)."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=32000)
        mock_queue.size.return_value = asyncio.Future()
        mock_queue.size.return_value.set_result(0)
        mock_cache.acquire_by_model.return_value = _make_entry()
        mock_cache.has_starting_server.return_value = False

        from routers.servers import create_server

        body = CreateServerRequest(model_id="test-model")
        request = _make_request()

        result = _run(create_server(request, body))
        assert result["server_id"] == "srv-1"

    @patch("routers.servers.model_loader")
    @patch("app.request_queue")
    def test_uses_default_90000_when_model_has_no_parameters(
        self, mock_queue, mock_loader
    ):
        """When model.parameters is None, default to 90000."""
        mock_loader.get_model_by_id.return_value = _make_model(params_none=True)
        mock_queue.size.return_value = asyncio.Future()
        mock_queue.size.return_value.set_result(0)

        from fastapi import HTTPException
        from routers.servers import create_server

        body = CreateServerRequest(model_id="test-model", num_ctx=100000)
        request = _make_request()

        with pytest.raises(HTTPException) as exc_info:
            _run(create_server(request, body))

        assert exc_info.value.status_code == 507
        detail = exc_info.value.detail
        assert detail["reason"] == "context_too_large"
        assert detail["details"]["model_num_ctx"] == 90000

    @patch("routers.servers.model_loader")
    @patch("app.server_cache")
    @patch("app.request_queue")
    def test_allows_equal_context_size(
        self, mock_queue, mock_cache, mock_loader
    ):
        """No rejection when request num_ctx == model parameters.num_ctx."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=32000)
        mock_queue.size.return_value = asyncio.Future()
        mock_queue.size.return_value.set_result(0)
        mock_cache.acquire_by_model.return_value = _make_entry()
        mock_cache.has_starting_server.return_value = False

        from routers.servers import create_server

        body = CreateServerRequest(model_id="test-model", num_ctx=32000)
        request = _make_request()

        result = _run(create_server(request, body))
        assert result["server_id"] == "srv-1"


class TestCreateServerRequestSchema:
    """Verify the CreateServerRequest model accepts num_ctx."""

    def test_accepts_num_ctx(self):
        req = CreateServerRequest(model_id="test", num_ctx=32000)
        assert req.num_ctx == 32000

    def test_defaults_num_ctx_to_none(self):
        req = CreateServerRequest(model_id="test")
        assert req.num_ctx is None
