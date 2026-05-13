"""Test handling of incomplete chunked reads in proxy router."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from httpx import RemoteProtocolError


@pytest.fixture
def mock_server_cache():
    """Provide a mock server_cache for proxy router tests."""
    mock_cache = MagicMock()
    mock_cache.decrement_use = MagicMock()
    with patch("app.server_cache", mock_cache):
        yield mock_cache


@pytest.mark.asyncio
async def test_stream_upstream_handles_incomplete_chunk(mock_server_cache):
    """Test that _stream_upstream handles incomplete chunked reads gracefully.

    _stream_upstream returns a StreamingResponse; the error only surfaces
    when the response body iterator is consumed. We must drain the iterator
    to trigger the mid-stream error.
    """
    from proxy.router import _stream_upstream

    mock_client = AsyncMock()

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    async def mock_iter():
        yield b"chunk1"
        raise RemoteProtocolError("peer closed connection")

    mock_response.aiter_bytes = mock_iter
    mock_response.aclose = AsyncMock()

    mock_client.send = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()

    response = await _stream_upstream(
        mock_client,
        "POST",
        "http://localhost/v1/chat/completions",
        {},
        b'{"stream": true}',
        "test-server",
    )

    # The error only surfaces when we consume the stream body
    with pytest.raises(RemoteProtocolError):
        async for _chunk in response.body_iterator:
            pass

    # Verify cleanup was called
    mock_response.aclose.assert_called_once()
    mock_client.aclose.assert_called_once()
    mock_server_cache.decrement_use.assert_called_once_with("test-server")


@pytest.mark.asyncio
async def test_stream_upstream_handles_generic_stream_error(mock_server_cache):
    """Test that _stream_upstream handles generic stream errors."""
    from proxy.router import _stream_upstream

    mock_client = AsyncMock()

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}

    async def mock_iter():
        yield b"chunk1"
        raise ValueError("stream error")

    mock_response.aiter_bytes = mock_iter
    mock_response.aclose = AsyncMock()

    mock_client.send = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()

    response = await _stream_upstream(
        mock_client,
        "POST",
        "http://localhost/v1/chat/completions",
        {},
        b'{"stream": true}',
        "test-server",
    )

    with pytest.raises(ValueError):
        async for _chunk in response.body_iterator:
            pass

    mock_response.aclose.assert_called_once()
    mock_client.aclose.assert_called_once()
    mock_server_cache.decrement_use.assert_called_once_with("test-server")


@pytest.mark.asyncio
async def test_stream_upstream_handles_disconnect_before_response(mock_server_cache):
    """Test that _stream_upstream handles upstream disconnect before any response.

    This covers the 'Server disconnected without sending a response' scenario
    from issue #28 — the upstream closes the TCP connection before sending
    HTTP response headers.
    """
    from proxy.router import _stream_upstream

    mock_client = AsyncMock()
    mock_client.send = AsyncMock(
        side_effect=RemoteProtocolError(
            "Server disconnected without sending a response"
        )
    )
    mock_client.aclose = AsyncMock()

    with pytest.raises(RemoteProtocolError):
        await _stream_upstream(
            mock_client,
            "POST",
            "http://localhost/v1/chat/completions",
            {},
            b'{"stream": true}',
            "test-server",
        )

    # Verify cleanup was called even on pre-response disconnect
    mock_client.aclose.assert_called_once()
    mock_server_cache.decrement_use.assert_called_once_with("test-server")


@pytest.mark.asyncio
async def test_stream_upstream_handles_connect_error_before_response(mock_server_cache):
    """Test that _stream_upstream handles connection errors before response."""
    from proxy.router import _stream_upstream

    mock_client = AsyncMock()
    mock_client.send = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    mock_client.aclose = AsyncMock()

    with pytest.raises(httpx.ConnectError):
        await _stream_upstream(
            mock_client,
            "POST",
            "http://localhost/v1/chat/completions",
            {},
            b'{"stream": true}',
            "test-server",
        )

    mock_client.aclose.assert_called_once()
    mock_server_cache.decrement_use.assert_called_once_with("test-server")


@pytest.mark.asyncio
async def test_stream_upstream_handles_timeout_before_response(mock_server_cache):
    """Test that _stream_upstream handles timeout errors before response."""
    from proxy.router import _stream_upstream

    mock_client = AsyncMock()
    mock_client.send = AsyncMock(
        side_effect=httpx.TimeoutException("Timeout")
    )
    mock_client.aclose = AsyncMock()

    with pytest.raises(httpx.TimeoutException):
        await _stream_upstream(
            mock_client,
            "POST",
            "http://localhost/v1/chat/completions",
            {},
            b'{"stream": true}',
            "test-server",
        )

    mock_client.aclose.assert_called_once()
    mock_server_cache.decrement_use.assert_called_once_with("test-server")
