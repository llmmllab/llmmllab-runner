"""Unit tests for the Hunyuan3D-Part (XPart) pipeline.

XPart's heavy deps (torch, spconv, partgen) aren't installed in the
test environment.  These tests exercise the lazy-load fence, payload
validation, registry discovery, and the file-serving endpoint using a
stand-in pipeline — same pattern as test_img23d_pipeline.py.
"""

import asyncio
import os
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from pipelines.mesh2parts.hunyuan3d_part import Hunyuan3DPartPipeline
from routers import pipelines as pipelines_router


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Lazy load gate
# ---------------------------------------------------------------------------


def test_hunyuan3d_part_lazy_load_friendly_error_when_deps_missing():
    """Missing partgen/torch surfaces a clean RuntimeError rather than
    propagating ImportError.  The error message points at the install
    recipe."""
    pipe = Hunyuan3DPartPipeline()
    # The real ``_load`` does ``from partgen.partformer_pipeline import
    # PartFormerPipeline`` — partgen isn't installed in tests, so this
    # path runs and raises.
    with pytest.raises(RuntimeError) as exc:
        _run(pipe._load())
    msg = str(exc.value)
    assert "Hunyuan3D-Part" in msg or "partgen" in msg


# ---------------------------------------------------------------------------
# _resolve_model_path — looks up details.model_path from .models.yaml
# ---------------------------------------------------------------------------


def test_resolve_model_path_reads_from_yaml(monkeypatch):
    """When no explicit path is passed, the pipeline looks up its own
    model_id in the ModelLoader registry and returns
    ``details.model_path``."""
    pipe = Hunyuan3DPartPipeline()

    fake_model = MagicMock()
    fake_model.details.model_path = "/models/hunyuan3d-part"

    fake_loader = MagicMock()
    fake_loader.get_model_by_id.return_value = fake_model

    monkeypatch.setattr(
        "utils.model_loader.ModelLoader", lambda: fake_loader
    )
    assert pipe._resolve_model_path() == "/models/hunyuan3d-part"
    fake_loader.get_model_by_id.assert_called_once_with("hunyuan3d-part")


def test_resolve_model_path_raises_when_yaml_missing(monkeypatch):
    pipe = Hunyuan3DPartPipeline()

    fake_loader = MagicMock()
    fake_loader.get_model_by_id.return_value = None

    monkeypatch.setattr(
        "utils.model_loader.ModelLoader", lambda: fake_loader
    )
    with pytest.raises(RuntimeError) as exc:
        pipe._resolve_model_path()
    assert "model registry" in str(exc.value)


def test_resolve_model_path_raises_when_model_path_unset(monkeypatch):
    pipe = Hunyuan3DPartPipeline()

    fake_model = MagicMock()
    fake_model.details.model_path = None

    fake_loader = MagicMock()
    fake_loader.get_model_by_id.return_value = fake_model

    monkeypatch.setattr(
        "utils.model_loader.ModelLoader", lambda: fake_loader
    )
    with pytest.raises(RuntimeError) as exc:
        pipe._resolve_model_path()
    assert "model_path" in str(exc.value)


# ---------------------------------------------------------------------------
# _run — payload validation runs BEFORE the model is invoked
# ---------------------------------------------------------------------------


def test_hunyuan3d_part_validates_missing_mesh():
    """``_run`` rejects payloads without ``mesh_b64`` even before
    invoking the underlying XPart instance."""
    pipe = Hunyuan3DPartPipeline()
    pipe._loaded = True
    pipe._impl = object()

    with pytest.raises(ValueError) as exc:
        _run(pipe._run({}))
    assert "mesh_b64 is required" in str(exc.value)


def test_hunyuan3d_part_validates_bad_base64():
    pipe = Hunyuan3DPartPipeline()
    pipe._loaded = True
    pipe._impl = object()

    with pytest.raises(ValueError) as exc:
        _run(pipe._run({"mesh_b64": "@@@not-base64@@@"}))
    assert "mesh_b64" in str(exc.value)


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------


def test_list_pipelines_includes_mesh2parts():
    listing = pipelines_router.list_pipelines()
    names = {p["name"] for p in listing["pipelines"]}
    assert "mesh2parts" in names

    entry = next(p for p in listing["pipelines"] if p["name"] == "mesh2parts")
    assert entry["task"] == "ImageTo3D"


# ---------------------------------------------------------------------------
# /v1/pipelines/mesh2parts/files/{filename}
# ---------------------------------------------------------------------------


def test_download_mesh2parts_artifact_serves_decomposed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipelines_router, "_IMG23D_PART_OUTPUT_DIR", str(tmp_path)
    )
    target = tmp_path / "abc123_decomposed.glb"
    target.write_bytes(b"glb-bytes")

    response = pipelines_router.download_mesh2parts_artifact(
        "abc123_decomposed.glb"
    )
    assert response.media_type == "model/gltf-binary"
    assert os.path.realpath(response.path) == os.path.realpath(str(target))


def test_download_mesh2parts_artifact_all_role_suffixes(tmp_path, monkeypatch):
    """All valid suffixes (decomposed/exploded/bbox/gt_bbox/input + the
    ``part_NN`` split form) are served; everything else 400s."""
    monkeypatch.setattr(
        pipelines_router, "_IMG23D_PART_OUTPUT_DIR", str(tmp_path)
    )

    for role in (
        "decomposed", "exploded", "bbox", "gt_bbox", "input",
        "part_00", "part_01", "part_99",
    ):
        f = tmp_path / f"abc_{role}.glb"
        f.write_bytes(b"x")
        resp = pipelines_router.download_mesh2parts_artifact(f.name)
        assert resp.media_type == "model/gltf-binary"


def test_download_mesh2parts_artifact_rejects_traversal(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(
        pipelines_router, "_IMG23D_PART_OUTPUT_DIR", str(tmp_path)
    )

    for bad in [
        "../etc/passwd",
        "../../secret.txt",
        "abc/def_decomposed.glb",
        "abc.exe",
        "abc_decomposed.glb.txt",
        # No-role suffix — must reject because we can't tell which
        # artefact the caller is after.
        "abc.glb",
        "",
        # Mismatched role labels — rejected.
        "abc_wrong.glb",
        # Bad part-NN forms — exactly 2 digits required.
        "abc_part_1.glb",
        "abc_part_100.glb",
        "abc_part_xx.glb",
        "abc_decompose.glb",
    ]:
        with pytest.raises(HTTPException) as exc:
            pipelines_router.download_mesh2parts_artifact(bad)
        assert exc.value.status_code == 400, f"expected 400 for {bad!r}"


def test_download_mesh2parts_artifact_404_when_missing(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(
        pipelines_router, "_IMG23D_PART_OUTPUT_DIR", str(tmp_path)
    )

    with pytest.raises(HTTPException) as exc:
        pipelines_router.download_mesh2parts_artifact(
            "missing_decomposed.glb"
        )
    assert exc.value.status_code == 404
