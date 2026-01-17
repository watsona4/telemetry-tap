"""Tests for Windows-specific collector functionality."""
from __future__ import annotations

import platform
from unittest.mock import Mock, patch, MagicMock
import pytest

from telemetry_tap.collector import MetricsCollector
from telemetry_tap.config import CollectorConfig, HealthConfig


@pytest.fixture
def mock_windows():
    """Mock Windows platform detection."""
    with patch("platform.system", return_value="Windows"):
        yield


@pytest.fixture
def collector_config():
    """Create a collector config for testing."""
    return CollectorConfig(
        smartctl_path="smartctl",
        lsblk_path="lsblk",
        sensors_path="sensors",
        dmidecode_path="dmidecode",
        apt_path="apt",
        dnf_path="dnf",
        systemctl_path="systemctl",
        librehardwaremonitor_url="http://localhost:8085/data.json",
        intel_gpu_top_path="intel_gpu_top",
        borg_path="borg",
        borg_repos=[],
        enable_tpu=False,
        enable_time_server=False,
        chronyc_path="chronyc",
        gpspipe_path="gpspipe",
        pps_device=None,
    )


@pytest.fixture
def health_config():
    """Create a health config for testing."""
    return HealthConfig(
        services=[],
        containers=[],
    )


@pytest.fixture
def collector(collector_config, health_config):
    """Create a MetricsCollector instance."""
    return MetricsCollector(collector_config, health_config)


@pytest.mark.windows
class TestWindowsCollector:
    """Test collector functionality on Windows."""

    def test_collector_init_on_windows(self, mock_windows, collector):
        """Test that collector initializes properly on Windows."""
        assert collector.config.librehardwaremonitor_url is not None
        assert collector.state is not None
        assert collector.state.drive_cache == {}

    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_collect_cpus_on_windows(self, mock_cpu_count, mock_cpu_percent, mock_windows, collector):
        """Test CPU collection on Windows."""
        # Mock cpu_percent to return list when percpu=True, float otherwise
        def cpu_percent_side_effect(interval=None, percpu=False):
            if percpu:
                return [10.0, 20.0, 30.0, 40.0]
            return 25.0

        mock_cpu_percent.side_effect = cpu_percent_side_effect
        mock_cpu_count.return_value = 2  # 2 physical cores, 4 logical

        cpus = collector._collect_cpus()

        assert len(cpus) == 1
        assert cpus[0]["name"] == "CPU"
        assert cpus[0]["num_physical_cores"] == 2
        assert cpus[0]["num_logical_cores"] == 4
        assert cpus[0]["load_pct"] == 25.0
        assert len(cpus[0]["cores"]) == 2

    @patch("psutil.virtual_memory")
    def test_collect_memory_on_windows(self, mock_vm, mock_windows, collector):
        """Test memory collection on Windows."""
        mock_vm.return_value = Mock(
            used=8589934592,  # 8GB
            available=8589934592,  # 8GB
            percent=50.0,
            total=17179869184,  # 16GB
        )

        memory = collector._collect_memory()

        assert memory["system"]["used_b"] == 8589934592
        assert memory["system"]["available_b"] == 8589934592
        assert memory["system"]["load_pct"] == 50.0

    @patch("psutil.disk_partitions")
    @patch("psutil.disk_usage")
    def test_collect_filesystems_on_windows(self, mock_usage, mock_partitions, mock_windows, collector):
        """Test filesystem collection on Windows."""
        mock_partitions.return_value = [
            Mock(
                device="C:\\",
                mountpoint="C:\\",
                fstype="NTFS",
            )
        ]
        mock_usage.return_value = Mock(
            used=107374182400,  # 100GB
            free=107374182400,  # 100GB
        )

        filesystems = collector._collect_filesystems()

        assert len(filesystems) == 1
        assert filesystems[0]["name"] == "C:\\"
        assert filesystems[0]["mountpoint"] == "C:\\"
        assert filesystems[0]["format"] == "NTFS"
        assert filesystems[0]["used_b"] == 107374182400
        assert filesystems[0]["available_b"] == 107374182400

    def test_collect_batteries_on_windows(self, mock_windows, collector):
        """Test battery collection on Windows (laptop scenario)."""
        with patch("psutil.sensors_battery") as mock_battery, \
             patch("telemetry_tap.collector.urlopen") as mock_urlopen:
            mock_battery.return_value = Mock(
                percent=75.0,
                power_plugged=False,
            )
            # Mock LHM to return no data
            mock_urlopen.side_effect = OSError("Connection refused")

            batteries = collector._collect_batteries()

            assert len(batteries) == 1
            assert batteries[0]["name"] == "Battery"
            assert batteries[0]["charge_level_pct"] == 75.0
            assert batteries[0]["discharging"] is True

    def test_collect_returns_valid_payload_on_windows(self, mock_windows, collector):
        """Test that collect() returns a valid payload structure on Windows."""
        def cpu_percent_side_effect(interval=None, percpu=False):
            if percpu:
                return [50.0, 50.0, 50.0, 50.0]
            return 50.0

        with patch("psutil.cpu_percent", side_effect=cpu_percent_side_effect), \
             patch("psutil.cpu_count", return_value=4), \
             patch("psutil.virtual_memory", return_value=Mock(used=1000, available=1000, percent=50.0, total=2000)), \
             patch("psutil.swap_memory", return_value=Mock(used=0, free=1000, percent=0.0)), \
             patch("psutil.boot_time", return_value=1000000), \
             patch("psutil.disk_partitions", return_value=[]), \
             patch("socket.gethostname", return_value="test-pc"):

            payload = collector.collect()

            # Verify required fields
            assert "schema" in payload
            assert payload["schema"]["name"] == "hwmon-exporter"
            assert payload["schema"]["version"] == 1
            assert "ts" in payload
            assert "host" in payload
            assert payload["host"]["name"] == "test-pc"
            assert "health" in payload
            assert "cpus" in payload
            assert "memory" in payload

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    @patch("psutil.disk_io_counters")
    def test_linux_tools_not_used_on_windows(self, mock_io_counters, mock_run_command, mock_windows, collector):
        """Test that Linux-specific tools are not called on Windows."""
        # Mock IO counters to provide drive data
        mock_io_counters.return_value = {
            "PhysicalDrive0": Mock(read_bytes=1000, write_bytes=2000)
        }
        # Mock _run_command to return None for smartctl (not available)
        mock_run_command.return_value = None

        collector._drive_metadata()

        # lsblk should not be called on Windows
        for call in mock_run_command.call_args_list:
            args = call[0][0] if call[0] else []
            assert "lsblk" not in str(args)

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_dmidecode_not_used_on_windows(self, mock_run_command, mock_windows, collector):
        """Test that dmidecode is not called on Windows."""
        collector._collect_dmidecode()

        # dmidecode should not be called on Windows
        mock_run_command.assert_not_called()

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_sensors_not_used_on_windows(self, mock_run_command, mock_windows, collector):
        """Test that sensors command is not called on Windows."""
        collector._collect_linux_sensors()

        # sensors should not be called on Windows
        mock_run_command.assert_not_called()


@pytest.mark.windows
class TestWindowsHealthChecks:
    """Test health check functionality on Windows."""

    @patch("subprocess.run")
    def test_service_checks_work_on_windows(self, mock_run, mock_windows, health_config, collector_config):
        """Test that Windows service checks work properly."""
        health = HealthConfig(services=["wuauserv"], containers=[])
        collector = MetricsCollector(collector_config, health)

        # Mock PowerShell response for Windows service query
        mock_run.return_value = Mock(
            returncode=0,
            stdout='[{"Name": "wuauserv", "Status": 4, "StartType": 2}]',
            stderr="",
        )

        services = collector._collect_services()

        # Should return Windows service status
        assert len(services) == 1
        assert services[0]["name"] == "wuauserv"
        assert services[0]["status"] == "active"
        assert services[0]["ok"] is True
        assert services[0]["loaded"] == "loaded"

    @patch("subprocess.run")
    def test_docker_container_checks_work_on_windows(self, mock_run, mock_windows, health_config, collector_config):
        """Test that Docker container checks work on Windows."""
        health = HealthConfig(services=[], containers=["test-container"])
        collector = MetricsCollector(collector_config, health)

        mock_run.return_value = Mock(
            returncode=0,
            stdout="test-container||Up 2 hours||nginx:latest",
            stderr="",
        )

        containers = collector._collect_containers()

        assert len(containers) == 1
        assert containers[0]["name"] == "test-container"
        assert containers[0]["status"] == "running"
        assert containers[0]["ok"] is True
