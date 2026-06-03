import subprocess
import time
from typing import Dict, List, Optional, Protocol

import nvsmi

from config import GPU_POWER_CAP_PCT
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="HardwareManager")


# ---------------------------------------------------------------------------
# Thermal throttling protocol + implementations
# ---------------------------------------------------------------------------

class ThermalThrottler(Protocol):
    """Protocol for architecture-specific thermal throttling.

    Implementations query temperatures and apply mitigation (e.g., power caps)
    when thresholds are exceeded.  New architectures (AMD ROCm, Intel, etc.)
    simply provide their own implementation.
    """

    def check_thermals(self) -> Dict[int, float]:
        """Check temperatures and apply mitigation when needed.

        Returns dict mapping device index → temperature in Celsius.
        """
        ...


class CudaThermalThrottler:
    """NVIDIA/CUDA thermal throttler using nvidia-smi."""

    def __init__(
        self,
        gpu_count: int,
        default_power_cap_pct: float,
        warning_c: float = 78.0,
        critical_c: float = 88.0,
        emergency_c: float = 92.0,
        critical_power_cap_pct: int = 50,
        emergency_power_cap_pct: int = 30,
    ):
        self._gpu_count = gpu_count
        self._default_power_cap_pct = default_power_cap_pct
        self._warning_c = warning_c
        self._critical_c = critical_c
        self._emergency_c = emergency_c
        self._critical_power_cap_pct = critical_power_cap_pct
        self._emergency_power_cap_pct = emergency_power_cap_pct
        self._last_critical_ts: Dict[int, float] = {}

    # -- public API ---------------------------------------------------------

    def check_thermals(self) -> Dict[int, float]:
        """Check GPU temperatures and apply thermal mitigation when needed.

        When a GPU reaches critical temperature, the power cap is aggressively
        reduced to prevent PCIe bus drop.  Once the GPU cools below the warning
        threshold, the original power cap is restored.

        Returns dict mapping device index → temperature in Celsius.
        """
        temps: Dict[int, float] = {}
        if self._gpu_count <= 0:
            return temps

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode != 0:
                return temps

            lines = result.stdout.strip().split("\n")
            for idx, line in enumerate(lines):
                try:
                    temp = float(line.strip())
                    temps[idx] = temp

                    if temp >= self._emergency_c:
                        logger.critical(
                            f"GPU {idx} EMERGENCY temperature: "
                            f"{temp}°C — throttling to {self._emergency_power_cap_pct}% power cap!"
                        )
                        if self._set_power_cap(idx, self._emergency_power_cap_pct):
                            self._last_critical_ts[idx] = time.time()
                    elif temp >= self._critical_c:
                        logger.error(
                            f"GPU {idx} CRITICAL temperature: "
                            f"{temp}°C — throttling to {self._critical_power_cap_pct}% power cap "
                            f"to prevent PCIe bus drop!"
                        )
                        if self._set_power_cap(idx, self._critical_power_cap_pct):
                            self._last_critical_ts[idx] = time.time()
                    elif temp >= self._warning_c:
                        logger.warning(
                            f"GPU {idx} high temperature: {temp}°C"
                        )
                    elif idx in self._last_critical_ts:
                        # GPU has cooled below warning — restore power cap
                        self._restore_power_cap(idx)
                        del self._last_critical_ts[idx]
                except ValueError:
                    continue
        except Exception as e:
            logger.debug(f"Thermal check failed: {e}")

        return temps

    # -- internal helpers ---------------------------------------------------

    def _set_power_cap(self, gpu_idx: int, pct: int) -> bool:
        """Set an absolute power cap on a single GPU (percentage of default limit).

        Returns True if the cap was successfully applied.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi", "-i", str(gpu_idx),
                    "--query-gpu=power.default_limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False

            default_watts = float(result.stdout.strip())
            target_watts = int(default_watts * pct / 100)

            set_result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_idx), "-pl", str(target_watts)],
                capture_output=True, text=True, timeout=10, check=False,
            )
            return set_result.returncode == 0
        except Exception:
            return False

    def _restore_power_cap(self, gpu_idx: int) -> None:
        """Restore the original power cap after thermal event cools down."""
        if self._default_power_cap_pct <= 0 or self._default_power_cap_pct > 100:
            return
        ok = self._set_power_cap(gpu_idx, int(self._default_power_cap_pct))
        if ok:
            logger.info(
                f"GPU {gpu_idx}: thermal event cleared, "
                f"power cap restored to {self._default_power_cap_pct:.0f}%"
            )


# ---------------------------------------------------------------------------
# HardwareManager
# ---------------------------------------------------------------------------

class HardwareManager:
    """GPU memory manager with power cap and thermal monitoring."""

    def __init__(self):
        self._has_gpu = False
        self._gpu_count = 0
        self._gpus: List[nvsmi.GPU] = []
        self._gpu_power_cap_pct = GPU_POWER_CAP_PCT
        self._thermal_throttler: Optional[CudaThermalThrottler] = None

        try:
            self._gpus = list(nvsmi.get_gpus())
            self._has_gpu = len(self._gpus) > 0
            self._gpu_count = len(self._gpus)
        except Exception:
            pass

        if self._has_gpu:
            self._apply_gpu_power_management()
            self._thermal_throttler = CudaThermalThrottler(
                gpu_count=self._gpu_count,
                default_power_cap_pct=self._gpu_power_cap_pct,
            )

    @property
    def has_gpu(self) -> bool:
        return self._has_gpu

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    def available_vram_bytes(self) -> float:
        """Total free VRAM across all GPUs, in bytes."""
        if not self._has_gpu:
            return 0.0
        try:
            self._gpus = list(nvsmi.get_gpus())
            return sum(g.mem_free for g in self._gpus) * 1024 * 1024
        except Exception:
            return 0.0

    def free_vram_by_gpu(self) -> Dict[int, float]:
        """Per-GPU free VRAM in bytes, keyed by integer device index.

        ``nvsmi`` reports GPUs in the same order CUDA enumerates them, so the
        list index aligns with the device index a llama.cpp ``tensor_split``
        refers to.  Returns an empty dict when no GPU is present.
        """
        out: Dict[int, float] = {}
        if not self._has_gpu:
            return out
        try:
            self._gpus = list(nvsmi.get_gpus())
            for idx, g in enumerate(self._gpus):
                out[idx] = float(g.mem_free) * 1024 * 1024
        except Exception:
            return {}
        return out

    @staticmethod
    def _parse_tensor_split(tensor_split):
        """Parse a llama.cpp ``tensor_split`` string into a weight list.

        ``"1,0,0"`` -> ``[1.0, 0.0, 0.0]`` (model only lands on device 0).
        Returns ``None`` for a missing/empty/malformed value — callers treat
        that as "no pinning, use total free VRAM across all GPUs".
        """
        if not tensor_split or not isinstance(tensor_split, str):
            return None
        try:
            return [float(x.strip()) for x in tensor_split.split(",") if x.strip()]
        except ValueError:
            return None

    def gpus_for_tensor_split(self, tensor_split) -> Optional[List[int]]:
        """Device indices a model with ``tensor_split`` will actually use.

        ``"1,0,0"`` -> ``[0]``.  Returns ``None`` when there's no pinning
        (use all GPUs).  Indices beyond the weight list (or with weight 0)
        are excluded.
        """
        weights = self._parse_tensor_split(tensor_split)
        if weights is None:
            return None
        return [i for i, w in enumerate(weights) if w > 0]

    def available_vram_bytes_for_split(self, tensor_split) -> float:
        """Free VRAM (bytes) only on the GPUs a ``tensor_split`` model lands on.

        Mirrors the api router's ``_effective_free_vram_bytes``: a model
        pinned to device 0 via ``tensor_split: "1,0,0"`` must NOT be credited
        with free VRAM that lives on devices 1 and 2.  This is the bug that
        made on-demand eviction a no-op for GPU-pinned models — the old
        ``available_vram_bytes()`` summed ALL cards, so a packed 12 GB GPU 0
        still looked like ~50 GB free (counting the two idle 3090s) and
        eviction returned early without freeing anything, letting the create
        OOM into a 500.  Falls back to total free VRAM when unpinned.
        """
        gpus = self.gpus_for_tensor_split(tensor_split)
        per_gpu = self.free_vram_by_gpu()
        if gpus is None:
            return sum(per_gpu.values()) if per_gpu else self.available_vram_bytes()
        return sum(per_gpu.get(i, 0.0) for i in gpus)

    def release_vram(self, *, log_before_after: bool = False) -> None:
        """Flush PyTorch's CUDA allocator cache back to the driver.

        Pipelines that have set their model handles to ``None`` still
        own VRAM until this is called — PyTorch's caching allocator
        holds freed-but-cached buffers and ``nvidia-smi`` keeps
        reporting them as resident.  Without flushing, back-to-back
        pipeline runs can OOM trying to load on a card whose
        driver-visible free VRAM is much smaller than the
        allocator's "we have plenty cached" view.

        Steps:
          1. ``gc.collect()`` — drop Python-side references that
             still anchor cuda tensors.
          2. ``torch.cuda.synchronize(i)`` per device — wait for
             in-flight kernels so their workspaces' refcounts
             actually fall to zero.
          3. ``torch.cuda.empty_cache()`` — release cached
             allocator blocks back to the driver.
          4. ``torch.cuda.ipc_collect()`` — drop any IPC tensors.

        Cheap (~50 ms) when there's nothing to release, so safe to
        call defensively before heavy loads.
        """
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch  # type: ignore[import-not-found]

            if not torch.cuda.is_available():
                return

            before: Optional[Dict[str, float]] = None
            if log_before_after:
                before = {
                    f"cuda:{i}": float(torch.cuda.mem_get_info(i)[0]) / 1024 ** 3
                    for i in range(torch.cuda.device_count())
                }

            # Synchronize + empty_cache PER DEVICE.  torch.cuda's
            # empty_cache only empties the allocator pool for the
            # CURRENT device (set via torch.cuda.set_device).  The
            # mesh2parts pipeline picks a primary GPU and shards
            # the DiT onto a secondary — without switching device
            # before each empty_cache, the secondary's allocator
            # pool keeps the ~7 GB DiT footprint cached even after
            # the module is gc'd.  That's the residue that's been
            # blocking llama-server tensor_split allocations on
            # the same card.
            original_device = None
            try:
                original_device = torch.cuda.current_device()
            except Exception:  # noqa: BLE001
                pass

            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(i)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

            if original_device is not None:
                try:
                    torch.cuda.set_device(original_device)
                except Exception:  # noqa: BLE001
                    pass

            try:
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001
                pass

            if log_before_after and before is not None:
                after = {
                    f"cuda:{i}": float(torch.cuda.mem_get_info(i)[0]) / 1024 ** 3
                    for i in range(torch.cuda.device_count())
                }
                logger.info(
                    "release_vram: "
                    + ", ".join(
                        f"{k} {before[k]:.1f}→{after[k]:.1f} GB free"
                        for k in before
                    )
                )
        except Exception:  # noqa: BLE001
            # Best-effort cleanup; never raise into the caller.
            pass

    def gpu_stats(self) -> Dict[str, Dict]:
        """Per-GPU stats: name, total_mb, used_mb, free_mb, util_percent, temperature_c."""
        stats: Dict[str, Dict] = {}
        if not self._has_gpu:
            return stats
        try:
            self._gpus = list(nvsmi.get_gpus())
            for g in self._gpus:
                stats[str(g.id)] = {
                    "name": g.name,
                    "total_mb": g.mem_total,
                    "used_mb": g.mem_used,
                    "free_mb": g.mem_free,
                    "util_percent": g.mem_util,
                }
        except Exception:
            pass
        return stats

    # ------------------------------------------------------------------
    # GPU power management (startup only)
    # ------------------------------------------------------------------

    def _apply_gpu_power_management(self) -> None:
        """Apply persistence mode and power cap to all GPUs on startup."""
        cap_pct = self._gpu_power_cap_pct
        if cap_pct <= 0 or cap_pct > 100:
            logger.info("GPU power cap disabled (GPU_POWER_CAP_PCT=0 or >100)")
            return

        for i in range(self._gpu_count):
            try:
                subprocess.run(
                    ["nvidia-smi", "-i", str(i), "-pm", "1"],
                    capture_output=True, text=True, timeout=10, check=False,
                )

                result = subprocess.run(
                    [
                        "nvidia-smi", "-i", str(i),
                        "--query-gpu=power.default_limit",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=10, check=False,
                )

                if result.returncode != 0 or not result.stdout.strip():
                    logger.debug(f"GPU {i}: could not query default power limit")
                    continue

                default_watts = float(result.stdout.strip())
                target_watts = int(default_watts * cap_pct / 100)

                set_result = subprocess.run(
                    ["nvidia-smi", "-i", str(i), "-pl", str(target_watts)],
                    capture_output=True, text=True, timeout=10, check=False,
                )

                if set_result.returncode == 0:
                    logger.info(
                        f"GPU {i}: power cap set to {target_watts}W "
                        f"({cap_pct:.0f}% of {default_watts:.0f}W default)"
                    )
                else:
                    logger.debug(
                        f"GPU {i}: could not set power cap — "
                        f"{set_result.stderr.strip()}"
                    )
            except Exception as e:
                logger.debug(f"GPU {i}: power management setup failed: {e}")

    # ------------------------------------------------------------------
    # Thermal monitoring (delegated to ThermalThrottler)
    # ------------------------------------------------------------------

    def check_gpu_thermals(self) -> Dict[int, float]:
        """Check GPU temperatures and apply thermal mitigation when needed.

        Delegates to the configured ThermalThrottler implementation.
        """
        if self._thermal_throttler is None:
            return {}
        return self._thermal_throttler.check_thermals()


hardware_manager = HardwareManager()
