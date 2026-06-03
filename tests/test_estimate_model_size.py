"""Tests for _estimate_model_size — VRAM footprint estimation used by
_evict_for_vram.

Regression context: the small-runner co-load OOM.  The estimate must
(a) include the mmproj/clip projector VRAM footprint (it loads to VRAM but
is not part of details.size), and (b) fall back sanely when details.size is
missing or the mmproj file can't be stat'd — so eviction frees enough VRAM
for two models not to co-load OOM on the single 12 GB card.
"""

import os

from unittest.mock import MagicMock

from routers.servers import _estimate_model_size

GB = 1024 * 1024 * 1024
MB = 1024 * 1024


def _model(size, clip_path=None):
    m = MagicMock()
    m.details.size = size
    m.details.clip_model_path = clip_path
    return m


class TestEstimateModelSize:
    def test_weights_only_adds_overhead(self):
        m = _model(2 * GB, clip_path=None)
        assert _estimate_model_size(m) == 2 * GB + 128 * MB

    def test_includes_mmproj_size(self, tmp_path):
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"\x00" * (3 * MB))  # tiny stand-in
        m = _model(2 * GB, clip_path=str(mmproj))
        est = _estimate_model_size(m)
        # weights + mmproj + overhead
        assert est == 2 * GB + 3 * MB + 128 * MB

    def test_missing_mmproj_file_uses_conservative_allowance(self):
        m = _model(2 * GB, clip_path="/does/not/exist/mmproj.gguf")
        est = _estimate_model_size(m)
        # weights + 1.5 GB mmproj allowance + overhead
        assert est == 2 * GB + int(1.5 * GB) + 128 * MB

    def test_zero_size_falls_back_to_4gb(self):
        m = _model(0, clip_path=None)
        assert _estimate_model_size(m) == 4 * GB

    def test_missing_size_attr_falls_back_to_4gb(self):
        m = MagicMock()
        # details.size raises when accessed
        type(m.details).size = property(
            lambda self: (_ for _ in ()).throw(AttributeError())
        )
        assert _estimate_model_size(m) == 4 * GB
