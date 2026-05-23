"""Unit tests for the in-process pipeline framework + TRELLIS pipeline.

TRELLIS' real dependencies (CUDA kernels, gsplat, custom rasterizers) are
not installed in the test environment.  These tests exercise the lazy-load
fence, the payload validation, and the router contract using a stand-in
pipeline.
"""

import asyncio
from typing import Any, Dict

import pytest

from models import ModelTask
from pipelines.base import InProcessPipeline
from pipelines.img23d.trellis import TrellisPipeline
from routers import pipelines as pipelines_router


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# InProcessPipeline contract
# ---------------------------------------------------------------------------


class _DummyPipeline(InProcessPipeline):
    name = "dummy"
    task = ModelTask.IMAGETO3D

    def __init__(self):
        super().__init__()
        self.load_calls = 0
        self.run_calls = 0

    async def _load(self):
        self.load_calls += 1

    async def _run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.run_calls += 1
        return {"echo": payload.get("value")}


def test_pipeline_loads_lazily_and_only_once():
    """Three concurrent ``run`` calls must trigger exactly one ``_load``."""
    pipe = _DummyPipeline()

    async def fire_three():
        return await asyncio.gather(
            pipe.run({"value": 1}),
            pipe.run({"value": 2}),
            pipe.run({"value": 3}),
        )

    results = _run(fire_three())
    assert pipe.load_calls == 1
    assert pipe.run_calls == 3
    assert {r["echo"] for r in results} == {1, 2, 3}


def test_pipeline_reloads_after_unload():
    pipe = _DummyPipeline()
    _run(pipe.run({"value": 1}))
    assert pipe.load_calls == 1

    _run(pipe.unload())
    assert pipe.loaded is False

    _run(pipe.run({"value": 2}))
    assert pipe.load_calls == 2


# ---------------------------------------------------------------------------
# TRELLIS pipeline — only the bits we can test without the real model.
# ---------------------------------------------------------------------------


def test_trellis_pipeline_declares_task():
    pipe = TrellisPipeline()
    assert pipe.name == "img23d"
    assert pipe.task == ModelTask.IMAGETO3D
    assert pipe.loaded is False


def test_trellis_load_raises_friendly_error_when_missing():
    """In CI we don't have the TRELLIS package — load() must explain that
    cleanly rather than blowing up with an opaque ImportError."""
    pipe = TrellisPipeline()
    with pytest.raises(RuntimeError) as exc:
        _run(pipe._load())
    assert "TRELLIS is not installed" in str(exc.value)


def test_trellis_validates_payload_missing_image():
    """``_run`` rejects payloads without ``image_b64`` even before invoking
    the underlying TRELLIS instance — we never want to load the model
    just to reject malformed input."""

    pipe = TrellisPipeline()
    # Bypass the lazy load gate: pretend the model is already loaded.
    pipe._loaded = True
    pipe._impl = object()

    with pytest.raises(ValueError) as exc:
        _run(pipe._run({}))
    assert "image_b64 is required" in str(exc.value)


def test_trellis_validates_payload_bad_base64():
    pipe = TrellisPipeline()
    pipe._loaded = True
    pipe._impl = object()

    with pytest.raises(ValueError) as exc:
        _run(pipe._run({"image_b64": "@@@not-base64@@@"}))
    assert "image_b64" in str(exc.value)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_list_pipelines_includes_img23d():
    listing = pipelines_router.list_pipelines()
    names = {p["name"] for p in listing["pipelines"]}
    assert "img23d" in names

    img23d_entry = next(p for p in listing["pipelines"] if p["name"] == "img23d")
    assert img23d_entry["task"] == "ImageTo3D"


def test_run_unknown_pipeline_returns_404():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _run(pipelines_router.run_pipeline("does-not-exist", {}))
    assert exc.value.status_code == 404
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail["reason"] == "pipeline_not_found"
    assert "img23d" in detail["available_pipelines"]
