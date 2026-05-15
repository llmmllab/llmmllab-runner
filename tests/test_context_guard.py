"""Tests for context-size guard in create_server endpoint.

Verifies that the runner rejects server creation when the requested
num_ctx exceeds the model's configured context window, returning 507
with a structured error before spinning up any llama.cpp process.
"""

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
    def test_rejects_when_requested_exceeds_model_ctx(
        self, mock_loader
    ):
        """507 when request num_ctx > model parameters.num_ctx."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=32000)

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
    def test_allows_when_requested_within_model_ctx(
        self, mock_cache, mock_loader
    ):
        """No rejection when request num_ctx <= model parameters.num_ctx."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=64000)
        mock_cache.acquire_by_model.return_value = _make_entry()
        mock_cache.has_starting_server.return_value = False

        from routers.servers import create_server

        body = CreateServerRequest(model_id="test-model", num_ctx=32000)
        request = _make_request()

        result = _run(create_server(request, body))
        assert result["server_id"] == "srv-1"

    @patch("routers.servers.model_loader")
    @patch("app.server_cache")
    def test_allows_when_num_ctx_not_provided(
        self, mock_cache, mock_loader
    ):
        """No rejection when request omits num_ctx (None)."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=32000)
        mock_cache.acquire_by_model.return_value = _make_entry()
        mock_cache.has_starting_server.return_value = False

        from routers.servers import create_server

        body = CreateServerRequest(model_id="test-model")
        request = _make_request()

        result = _run(create_server(request, body))
        assert result["server_id"] == "srv-1"

    @patch("routers.servers.model_loader")
    def test_uses_default_90000_when_model_has_no_parameters(
        self, mock_loader
    ):
        """When model.parameters is None, default to 90000."""
        mock_loader.get_model_by_id.return_value = _make_model(params_none=True)

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
    def test_allows_equal_context_size(
        self, mock_cache, mock_loader
    ):
        """No rejection when request num_ctx == model parameters.num_ctx."""
        mock_loader.get_model_by_id.return_value = _make_model(ctx=32000)
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
