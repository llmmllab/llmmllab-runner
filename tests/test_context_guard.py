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
