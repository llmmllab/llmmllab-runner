import subprocess
from typing import Dict, List

import nvsmi

from config import GPU_POWER_CAP_PCT
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="HardwareManager")


class HardwareManager:
    """GPU memory manager with power cap and thermal monitoring."""

    def __init__(self):
        self._has_gpu = False
        self._gpu_count = 0
        self._gpus: List[nvsmi.GPU] = []
        self._gpu_power_cap_pct = GPU_POWER_CAP_PCT
        self._thermal_warning_c = 78.0
        self._thermal_critical_c = 88.0
        try:
            self._gpus = list(nvsmi.get_gpus())
            self._has_gpu = len(self._gpus) > 0
            self._gpu_count = len(self._gpus)
        except Exception:
            pass

        if self._has_gpu:
            self._apply_gpu_power_management()

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
    # GPU thermal / power management
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

    def check_gpu_thermals(self) -> Dict[int, float]:
        """Check GPU temperatures and log warnings for hot devices.

        Returns dict mapping device index → temperature in Celsius.
        """
        temps: Dict[int, float] = {}
        if not self._has_gpu:
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

                    if temp >= self._thermal_critical_c:
                        logger.error(
                            f"GPU {idx} CRITICAL temperature: "
                            f"{temp}°C — risk of PCIe bus drop!"
                        )
                    elif temp >= self._thermal_warning_c:
                        logger.warning(
                            f"GPU {idx} high temperature: {temp}°C"
                        )
                except ValueError:
                    continue
        except Exception as e:
            logger.debug(f"Thermal check failed: {e}")

        return temps


hardware_manager = HardwareManager()
