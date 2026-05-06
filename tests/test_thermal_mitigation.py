"""Tests for GPU thermal mitigation in HardwareManager."""

from unittest.mock import MagicMock, patch

from utils.hardware_manager import HardwareManager


class TestThermalMitigation:
    """Verify that check_gpu_thermals actively mitigates hot GPUs."""

    @patch("utils.hardware_manager.nvsmi")
    def _make_manager(self, mock_nvsmi, gpu_count=1):
        mock_nvsmi.get_gpus.return_value = [
            MagicMock(id=i, name="Test GPU", mem_total=24576,
                      mem_used=0, mem_free=24576, mem_util=0)
            for i in range(gpu_count)
        ]
        with patch("utils.hardware_manager.GPU_POWER_CAP_PCT", 0):
            return HardwareManager()

    @patch("utils.hardware_manager.subprocess.run")
    def test_critical_temp_triggers_power_throttle(self, mock_run):
        """At 88°C the GPU power cap should be aggressively reduced."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="88.0\n", stderr=""
        )
        mgr = self._make_manager()
        temps = mgr.check_gpu_thermals()

        assert temps[0] == 88.0
        # _set_power_cap calls subprocess.run twice:
        # 1. query default_limit  2. set -pl
        assert mock_run.call_count >= 2
        # The last call should be setting the power limit
        set_call = mock_run.call_args_list[-1]
        assert "-pl" in set_call[0][0]

    @patch("utils.hardware_manager.subprocess.run")
    def test_emergency_temp_triggers_extreme_throttle(self, mock_run):
        """At 92°C the GPU should be throttled even harder."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="92.0\n", stderr=""
        )
        mgr = self._make_manager()
        temps = mgr.check_gpu_thermals()

        assert temps[0] == 92.0
        set_call = mock_run.call_args_list[-1]
        assert "-pl" in set_call[0][0]

    @patch("utils.hardware_manager.subprocess.run")
    def test_warning_temp_does_not_throttle(self, mock_run):
        """At 78°C only a warning is logged, no power cap change."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="78.0\n", stderr=""
        )
        mgr = self._make_manager()
        temps = mgr.check_gpu_thermals()

        assert temps[0] == 78.0
        # Only the temperature query call, no power cap calls
        assert mock_run.call_count == 1

    @patch("utils.hardware_manager.subprocess.run")
    def test_normal_temp_does_nothing(self, mock_run):
        """Below warning threshold, no action taken."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="55.0\n", stderr=""
        )
        mgr = self._make_manager()
        temps = mgr.check_gpu_thermals()

        assert temps[0] == 55.0
        assert mock_run.call_count == 1

    @patch("utils.hardware_manager.subprocess.run")
    def test_cool_down_restores_power_cap(self, mock_run):
        """After a critical event, cooling below warning restores power cap."""
        mgr = self._make_manager()
        mgr._gpu_power_cap_pct = 85

        # First call: critical temp
        mock_run.return_value = MagicMock(
            returncode=0, stdout="88.0\n", stderr=""
        )
        mgr.check_gpu_thermals()
        # _last_critical_ts should be populated
        assert 0 in mgr._last_critical_ts

        # Second call: cooled down
        mock_run.return_value = MagicMock(
            returncode=0, stdout="60.0\n", stderr=""
        )
        mgr.check_gpu_thermals()
        # Power cap should be restored, tracking cleared
        assert 0 not in mgr._last_critical_ts
