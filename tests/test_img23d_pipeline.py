"""Unit tests for the in-process pipeline framework + Hunyuan3D pipeline.

Hunyuan3D-2.1's real dependencies (torch + custom_rasterizer +
differentiable_renderer CUDA extensions) aren't installed in the test
environment.  These tests exercise the lazy-load fence, the payload
validation, and the router contract using a stand-in pipeline.
"""

import asyncio
from typing import Any, Dict

import pytest

from models import ModelTask
from pipelines.base import InProcessPipeline
from pipelines.img23d.hunyuan3d import Hunyuan3DPipeline
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
# Hunyuan3D pipeline — only the bits we can test without the real model.
# ---------------------------------------------------------------------------


def test_hunyuan3d_pipeline_declares_task():
    pipe = Hunyuan3DPipeline()
    assert pipe.name == "img23d"
    assert pipe.task == ModelTask.IMAGETO3D
    assert pipe.loaded is False


def test_hunyuan3d_load_raises_friendly_error_when_missing():
    """In CI we don't have the hy3dgen package — load() must explain that
    cleanly rather than blowing up with an opaque ImportError."""
    pipe = Hunyuan3DPipeline()
    with pytest.raises(RuntimeError) as exc:
        _run(pipe._load())
    assert "Hunyuan3D" in str(exc.value)


def test_hunyuan3d_validates_payload_missing_image():
    """``_run`` rejects payloads without ``image_b64`` even before invoking
    the underlying Hunyuan3D instance — we never want to load the model
    just to reject malformed input."""

    pipe = Hunyuan3DPipeline()
    # Bypass the lazy load gate: pretend the model is already loaded.
    pipe._loaded = True
    pipe._impl = object()

    with pytest.raises(ValueError) as exc:
        _run(pipe._run({}))
    assert "image_b64 is required" in str(exc.value)


def test_hunyuan3d_validates_payload_bad_base64():
    pipe = Hunyuan3DPipeline()
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


# ---------------------------------------------------------------------------
# /v1/pipelines/img23d/files/{filename}
# ---------------------------------------------------------------------------


def test_download_img23d_artifact_serves_existing_file(tmp_path, monkeypatch):
    """Files inside the configured output dir are served verbatim with the
    right media type."""
    monkeypatch.setattr(pipelines_router, "_IMG23D_OUTPUT_DIR", str(tmp_path))
    target = tmp_path / "abc123.glb"
    target.write_bytes(b"glb-bytes")

    response = pipelines_router.download_img23d_artifact("abc123.glb")
    assert response.media_type == "model/gltf-binary"
    # FileResponse.path is the absolute path it'll serve from.
    assert os.path.realpath(response.path) == os.path.realpath(str(target))


def test_download_img23d_artifact_rejects_traversal(tmp_path, monkeypatch):
    """Any filename with path separators or unexpected characters must 400."""
    from fastapi import HTTPException

    monkeypatch.setattr(pipelines_router, "_IMG23D_OUTPUT_DIR", str(tmp_path))

    for bad in [
        "../etc/passwd",
        "../../secret.txt",
        "abc/def.glb",
        "abc.exe",
        "abc.glb.txt",
        "",
    ]:
        with pytest.raises(HTTPException) as exc:
            pipelines_router.download_img23d_artifact(bad)
        assert exc.value.status_code == 400, f"expected 400 for {bad!r}"


def test_download_img23d_artifact_returns_404_for_missing(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(pipelines_router, "_IMG23D_OUTPUT_DIR", str(tmp_path))

    with pytest.raises(HTTPException) as exc:
        pipelines_router.download_img23d_artifact("does-not-exist.glb")
    assert exc.value.status_code == 404


def test_download_img23d_artifact_supports_ply_and_png(tmp_path, monkeypatch):
    monkeypatch.setattr(pipelines_router, "_IMG23D_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "a.ply").write_bytes(b"ply")
    (tmp_path / "b.png").write_bytes(b"png")

    assert pipelines_router.download_img23d_artifact("a.ply").media_type == "application/octet-stream"
    assert pipelines_router.download_img23d_artifact("b.png").media_type == "image/png"


# Need the os import for the test above.
import os  # noqa: E402
