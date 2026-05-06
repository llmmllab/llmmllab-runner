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
