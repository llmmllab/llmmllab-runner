"""Tests for on-demand VRAM-pressure eviction.

Regression context: under VRAM pressure, ``/v1/server/create`` used to 500
(cudaMalloc OOM, tripping the api circuit breaker) because the eviction path
only freed servers idle past CACHE_TIMEOUT_MIN (10 min).  A server that JUST
went idle (use_count == 0) couldn't be reclaimed, so a brand-new model could
never get a slot.

These tests cover:
  * ``ServerCache.get_idle_lru_for_vram`` — the LRU/idle/min-residency
    candidate selection (never returns busy or too-recently-idle servers);
  * ``_evict_for_vram`` — evicts coldest-first until the model fits, waits for
    the driver to reclaim VRAM, and reports False when it can't free enough.
"""

import time
from unittest.mock import MagicMock, patch

from cache import ServerCache, ServerEntry

GB = 1024 * 1024 * 1024
MB = 1024 * 1024


def _entry(server_id, *, use_count=0, idle_since=None, starting=False):
    return ServerEntry(
        server_id=server_id,
        model_id=f"m-{server_id}",
        port=8001,
        use_count=use_count,
        idle_since=idle_since,
        starting=starting,
        manager=None,
    )


class TestGetIdleLruForVram:
    def _cache_with(self, entries):
        c = ServerCache()
        for e in entries:
            c._servers[e.server_id] = e
        return c

    def test_returns_idle_past_min_residency_lru_first(self):
        now = time.time()
        c = self._cache_with([
            _entry("new", idle_since=now - 10),     # within min-residency
            _entry("old", idle_since=now - 600),    # coldest
            _entry("mid", idle_since=now - 120),
        ])
        out = c.get_idle_lru_for_vram(min_residency_sec=30)
        ids = [e.server_id for e in out]
        # "new" excluded (idle < 30 s); coldest first.
        assert ids == ["old", "mid"]

    def test_excludes_busy_servers(self):
        now = time.time()
        c = self._cache_with([
            _entry("busy", use_count=1, idle_since=None),
            _entry("busy2", use_count=2, idle_since=now - 600),  # use_count wins
            _entry("idle", use_count=0, idle_since=now - 600),
        ])
        out = c.get_idle_lru_for_vram(min_residency_sec=30)
        assert [e.server_id for e in out] == ["idle"]

    def test_excludes_starting_servers(self):
        now = time.time()
        c = self._cache_with([
            _entry("starting", idle_since=now - 600, starting=True),
            _entry("idle", idle_since=now - 600),
        ])
        out = c.get_idle_lru_for_vram(min_residency_sec=30)
        assert [e.server_id for e in out] == ["idle"]

    def test_excludes_never_idle(self):
        c = self._cache_with([
            _entry("noidle", use_count=0, idle_since=None),
        ])
        assert c.get_idle_lru_for_vram(min_residency_sec=30) == []

    def test_zero_min_residency_includes_just_idle(self):
        now = time.time()
        c = self._cache_with([_entry("fresh", idle_since=now - 0.001)])
        out = c.get_idle_lru_for_vram(min_residency_sec=0)
        assert [e.server_id for e in out] == ["fresh"]


def _model(size_gb, tensor_split=None):
    m = MagicMock()
    m.id = "Qwen3_6_27B"
    m.details.size = int(size_gb * GB)
    m.details.clip_model_path = None
    m.parameters.tensor_split = tensor_split
    return m


class TestEvictForVram:
    """``_evict_for_vram`` integration over a real ServerCache + mocked
    hardware_manager / app.server_cache module wiring."""

    def _install_cache(self, entries):
        """Patch ``app.server_cache`` (imported lazily inside the function)
        with a ServerCache preloaded with *entries*, each given a stop()able
        manager mock.  Returns (cache, managers)."""
        cache = ServerCache()
        managers = {}
        for e in entries:
            mgr = MagicMock()
            mgr._intentional_stop = False
            e.manager = mgr
            managers[e.server_id] = mgr
            cache._servers[e.server_id] = e
        return cache, managers

    def test_no_pressure_returns_true_without_evicting(self):
        cache, managers = self._install_cache([
            _entry("a", idle_since=time.time() - 600),
        ])
        from routers import servers as servers_mod
        import app
        app.server_cache = cache

        with patch.object(
            servers_mod.hardware_manager, "available_vram_bytes_for_split",
            return_value=30 * GB,
        ):
            assert servers_mod._evict_for_vram(_model(10)) is True
        managers["a"].stop.assert_not_called()

    def test_evicts_coldest_until_fits(self):
        now = time.time()
        cache, managers = self._install_cache([
            _entry("cold", idle_since=now - 600),
            _entry("warm", idle_since=now - 60),
        ])
        from routers import servers as servers_mod

        # Free VRAM: starts at 2 GB; after the FIRST eviction jumps to 12 GB
        # (enough for a 10 GB model), so only "cold" should be stopped.
        vram = iter([2 * GB] + [12 * GB] * 20)
        last = {"v": 2 * GB}

        def _avail(*_a, **_k):
            try:
                last["v"] = next(vram)
            except StopIteration:
                pass
            return last["v"]

        import app
        app.server_cache = cache
        with patch.object(
            servers_mod.hardware_manager, "available_vram_bytes_for_split",
            side_effect=_avail,
        ):
            ok = servers_mod._evict_for_vram(_model(10))
        assert ok is True
        managers["cold"].stop.assert_called_once()
        managers["warm"].stop.assert_not_called()
        assert "cold" not in cache._servers
        assert "warm" in cache._servers
        # Eviction must be marked intentional so the watchdog stays quiet.
        assert managers["cold"]._intentional_stop is True

    def test_returns_false_when_cannot_free_enough(self):
        now = time.time()
        cache, managers = self._install_cache([
            _entry("cold", idle_since=now - 600),
        ])
        from routers import servers as servers_mod
        import app
        app.server_cache = cache

        # Free VRAM never reaches the 10 GB needed even after eviction.
        with patch.object(
            servers_mod.hardware_manager, "available_vram_bytes_for_split",
            return_value=1 * GB,
        ):
            ok = servers_mod._evict_for_vram(_model(10))
        assert ok is False
        managers["cold"].stop.assert_called_once()  # tried, freed nothing

    def test_returns_false_when_no_idle_candidates(self):
        # All servers busy → nothing to evict → False, nothing stopped.
        cache, managers = self._install_cache([
            _entry("busy", use_count=1, idle_since=None),
        ])
        from routers import servers as servers_mod
        import app
        app.server_cache = cache

        with patch.object(
            servers_mod.hardware_manager, "available_vram_bytes_for_split",
            return_value=1 * GB,
        ):
            ok = servers_mod._evict_for_vram(_model(10))
        assert ok is False
        managers["busy"].stop.assert_not_called()

    def test_tensor_split_aware_does_not_evict_for_other_gpu_pressure(self):
        """Regression for the live failure: a GPU-0-pinned model
        (tensor_split "1,0,0") under pressure on GPU 0 must measure free VRAM
        ON GPU 0 only.  With per-split accounting reporting 2 GB free on GPU 0
        (need 10 GB), eviction must fire — even though total free VRAM across
        the idle 3090s would look like plenty under the old all-GPU sum.
        """
        now = time.time()
        cache, managers = self._install_cache([
            _entry("cold", idle_since=now - 600),
        ])
        from routers import servers as servers_mod
        import app
        app.server_cache = cache

        # Per-split free VRAM: 2 GB before eviction, 12 GB after.
        vram = iter([2 * GB, 2 * GB, 12 * GB, 12 * GB, 12 * GB])
        last = {"v": 2 * GB}

        def _avail(_ts):
            try:
                last["v"] = next(vram)
            except StopIteration:
                pass
            return last["v"]

        with patch.object(
            servers_mod.hardware_manager, "available_vram_bytes_for_split",
            side_effect=_avail,
        ):
            ok = servers_mod._evict_for_vram(_model(10, tensor_split="1,0,0"))
        assert ok is True
        managers["cold"].stop.assert_called_once()


class TestHardwareManagerTensorSplit:
    """Tensor-split-aware VRAM accounting helpers on HardwareManager."""

    def _hm(self, per_gpu_free_mb):
        import sys
        from utils.hardware_manager import HardwareManager

        hwmod = sys.modules["utils.hardware_manager"]
        hm = HardwareManager.__new__(HardwareManager)
        hm._has_gpu = True

        class _G:
            def __init__(self, free_mb):
                self.mem_free = free_mb

        gpus = [_G(mb) for mb in per_gpu_free_mb]
        hm._gpus = gpus
        # free_vram_by_gpu / available_vram_bytes re-read via nvsmi.get_gpus();
        # patch the module-level nvsmi so they see our fake GPUs.
        self._nvsmi_patch = patch.object(
            hwmod.nvsmi, "get_gpus", return_value=gpus
        )
        self._nvsmi_patch.start()
        return hm

    def teardown_method(self):
        p = getattr(self, "_nvsmi_patch", None)
        if p is not None:
            p.stop()

    def test_gpus_for_tensor_split_pins_to_device_0(self):
        hm = self._hm([12000, 24000, 24000])
        assert hm.gpus_for_tensor_split("1,0,0") == [0]

    def test_gpus_for_tensor_split_multi(self):
        hm = self._hm([12000, 24000, 24000])
        assert hm.gpus_for_tensor_split("0,1,1") == [1, 2]

    def test_gpus_for_tensor_split_none_unpinned(self):
        hm = self._hm([12000, 24000, 24000])
        assert hm.gpus_for_tensor_split(None) is None

    def test_available_for_split_counts_only_pinned_gpu(self):
        # 3 GB free on GPU 0, 23 GB on each 3090.
        hm = self._hm([3000, 23000, 23000])
        eff = hm.available_vram_bytes_for_split("1,0,0")
        # Only GPU 0 counts → ~3 GB, NOT ~49 GB.
        assert abs(eff - 3000 * MB) < MB

    def test_available_for_split_unpinned_sums_all(self):
        hm = self._hm([3000, 23000, 23000])
        eff = hm.available_vram_bytes_for_split(None)
        assert abs(eff - (3000 + 23000 + 23000) * MB) < MB
