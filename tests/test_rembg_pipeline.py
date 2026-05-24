"""Unit tests for the rembg (briaai/RMBG-2.0) pipeline.

transformers + torch + the model weights aren't installed in the CI
environment.  We test the lazy-load fence, the payload validation, and
the file-serving endpoint contract — the same pattern test_img23d_
pipeline.py uses for Hunyuan3D.
"""

import asyncio
import os
import pytest

from models import ModelTask
from pipelines.rembg.rmbg import RMBGPipeline
from routers import pipelines as pipelines_router


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_rembg_pipeline_declares_image_to_image_task():
    pipe = RMBGPipeline()
    assert pipe.name == "rembg"
    assert pipe.task == ModelTask.IMAGETOIMAGE
    assert pipe.loaded is False


def test_rembg_load_raises_friendly_error_when_missing():
    pipe = RMBGPipeline()
    # transformers IS installed on dev machines, so the load may
    # succeed.  We only assert error behaviour when the import actually
    # fails — i.e., when transformers/torch are absent.  In that case
    # the message must mention RMBG-2.0 / transformers so an operator
    # knows what to install.
    try:
        import transformers  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import torchvision  # type: ignore[import-not-found]
        pytest.skip("dependencies present; missing-dep path is irrelevant here")
    except ImportError:
        with pytest.raises(RuntimeError) as exc:
            _run(pipe._load())
        msg = str(exc.value).lower()
        assert "rmbg" in msg or "transformers" in msg


def test_rembg_validates_payload_missing_image():
    pipe = RMBGPipeline()
    pipe._loaded = True
    pipe._impl = object()
    with pytest.raises(ValueError) as exc:
        _run(pipe._run({}))
    assert "image_b64 is required" in str(exc.value)


def test_rembg_validates_payload_bad_base64():
    pipe = RMBGPipeline()
    pipe._loaded = True
    pipe._impl = object()
    with pytest.raises(ValueError) as exc:
        _run(pipe._run({"image_b64": "@@@not-base64@@@"}))
    assert "image_b64" in str(exc.value)


def test_list_pipelines_includes_rembg():
    listing = pipelines_router.list_pipelines()
    names = {p["name"] for p in listing["pipelines"]}
    assert "rembg" in names

    rembg_entry = next(p for p in listing["pipelines"] if p["name"] == "rembg")
    assert rembg_entry["task"] == "ImageToImage"


# ---------------------------------------------------------------------------
# File-serving endpoint
# ---------------------------------------------------------------------------


def test_download_rembg_artifact_serves_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pipelines_router, "_REMBG_OUTPUT_DIR", str(tmp_path))
    target = tmp_path / "abc123.png"
    target.write_bytes(b"png-bytes")

    response = pipelines_router.download_rembg_artifact("abc123.png")
    assert response.media_type == "image/png"
    assert os.path.realpath(response.path) == os.path.realpath(str(target))


def test_download_rembg_artifact_rejects_traversal(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(pipelines_router, "_REMBG_OUTPUT_DIR", str(tmp_path))
    for bad in ["../etc/passwd", "abc/def.png", "abc.exe", "abc.png.txt", ""]:
        with pytest.raises(HTTPException) as exc:
            pipelines_router.download_rembg_artifact(bad)
        assert exc.value.status_code == 400, f"expected 400 for {bad!r}"


def test_download_rembg_artifact_404_when_missing(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(pipelines_router, "_REMBG_OUTPUT_DIR", str(tmp_path))
    with pytest.raises(HTTPException) as exc:
        pipelines_router.download_rembg_artifact("does-not-exist.png")
    assert exc.value.status_code == 404
