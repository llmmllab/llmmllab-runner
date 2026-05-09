"""Tests for improved error messaging in server and model routers."""

import pytest
from unittest.mock import MagicMock, patch

from routers.servers import (
    _build_error_response,
    _get_available_model_ids,
    MODEL_NOT_CONFIGURED,
    MODEL_NOT_AVAILABLE,
    INSUFFICIENT_RESOURCES,
    SERVER_START_FAILED,
)
from routers.models import get_model


class TestBuildErrorResponse:
    def test_minimal_response(self):
        result = _build_error_response(reason="test", message="Test message")
        assert result == {"reason": "test", "message": "Test message"}

    def test_with_requested_model(self):
        result = _build_error_response(
            reason="test", message="Test", requested_model="gpt-4"
        )
        assert result["requested_model"] == "gpt-4"

    def test_with_available_models(self):
        result = _build_error_response(
            reason="test", message="Test", available_models=["m1", "m2"]
        )
        assert result["available_models"] == ["m1", "m2"]

    def test_with_details(self):
        result = _build_error_response(
            reason="test",
            message="Test",
            details={"exit_code": -9, "vram_bytes": 1024},
        )
        assert result["details"]["exit_code"] == -9

    def test_full_response(self):
        result = _build_error_response(
            reason=MODEL_NOT_CONFIGURED,
            message="Model 'gpt-4' not configured",
            requested_model="gpt-4",
            available_models=["llama-3", "mistral"],
            details={"runner": "runner-small"},
        )
        assert result["reason"] == MODEL_NOT_CONFIGURED
        assert result["requested_model"] == "gpt-4"
        assert result["available_models"] == ["llama-3", "mistral"]
        assert result["details"]["runner"] == "runner-small"


class TestGetAvailableModelIds:
    @patch("routers.servers.model_loader")
    def test_returns_sorted_ids(self, mock_loader):
        mock_loader.get_available_models.return_value = {
            "z-model": MagicMock(),
            "a-model": MagicMock(),
            "m-model": MagicMock(),
        }
        result = _get_available_model_ids()
        assert result == ["a-model", "m-model", "z-model"]

    @patch("routers.servers.model_loader")
    def test_empty_when_no_models(self, mock_loader):
        mock_loader.get_available_models.return_value = {}
        result = _get_available_model_ids()
        assert result == []


class TestErrorReasonCodes:
    def test_reason_codes_are_strings(self):
        assert isinstance(MODEL_NOT_CONFIGURED, str)
        assert isinstance(MODEL_NOT_AVAILABLE, str)
        assert isinstance(INSUFFICIENT_RESOURCES, str)
        assert isinstance(SERVER_START_FAILED, str)

    def test_reason_codes_are_distinct(self):
        codes = {
            MODEL_NOT_CONFIGURED,
            MODEL_NOT_AVAILABLE,
            INSUFFICIENT_RESOURCES,
            SERVER_START_FAILED,
        }
        assert len(codes) == 4  # all distinct


class TestGetModelEndpoint:
    @patch("routers.models.model_loader")
    def test_returns_model_when_found(self, mock_loader):
        mock_model = MagicMock()
        mock_model.id = "test-model"
        mock_model.name = "Test Model"
        mock_model.provider = "test-provider"
        mock_model.task.value = "TextToText"
        mock_model.details = None
        mock_loader.get_model_by_id.return_value = mock_model

        result = get_model("test-model")
        assert result["model_id"] == "test-model"
        assert result["name"] == "Test Model"

    @patch("routers.models.model_loader")
    def test_returns_404_with_structured_error(self, mock_loader):
        mock_loader.get_model_by_id.return_value = None
        mock_loader.get_available_models.return_value = {
            "model-a": MagicMock(),
            "model-b": MagicMock(),
        }

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            get_model("nonexistent-model")

        assert exc_info.value.status_code == 404
        detail = exc_info.value.detail
        assert detail["reason"] == "model_not_configured"
        assert "nonexistent-model" in detail["message"]
        assert detail["requested_model"] == "nonexistent-model"
        assert detail["available_models"] == ["model-a", "model-b"]
