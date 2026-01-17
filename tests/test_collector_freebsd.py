"""Tests for FreeBSD and OPNsense-specific collector functionality."""
from __future__ import annotations

from unittest.mock import Mock, patch
import pytest

from telemetry_tap.collector import MetricsCollector
from telemetry_tap.config import CollectorConfig, HealthConfig


@pytest.fixture
def mock_freebsd():
    """Mock FreeBSD platform detection."""
    with patch("platform.system", return_value="FreeBSD"):
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
        librehardwaremonitor_url=None,
        intel_gpu_top_path="intel_gpu_top",
        borg_path="borg",
        borg_repos=[],
        enable_tpu=False,
        enable_time_server=False,
        chronyc_path="chronyc",
        gpspipe_path="gpspipe",
        pps_device=None,
        sysctl_path="sysctl",
        pkg_path="pkg",
        service_path="service",
        pluginctl_path="pluginctl",
        configctl_path="configctl",
        zenarmorctl_path="zenarmorctl",
        enable_opnsense=False,
        enable_zenarmor=False,
    )


@pytest.fixture
def collector_config_opnsense():
    """Create a collector config with OPNsense enabled."""
    return CollectorConfig(
        smartctl_path="smartctl",
        lsblk_path="lsblk",
        sensors_path="sensors",
        dmidecode_path="dmidecode",
        apt_path="apt",
        dnf_path="dnf",
        systemctl_path="systemctl",
        librehardwaremonitor_url=None,
        intel_gpu_top_path="intel_gpu_top",
        borg_path="borg",
        borg_repos=[],
        enable_tpu=False,
        enable_time_server=False,
        chronyc_path="chronyc",
        gpspipe_path="gpspipe",
        pps_device=None,
        sysctl_path="sysctl",
        pkg_path="pkg",
        service_path="service",
        pluginctl_path="pluginctl",
        configctl_path="configctl",
        zenarmorctl_path="zenarmorctl",
        enable_opnsense=True,
        enable_zenarmor=True,
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


@pytest.fixture
def collector_opnsense(collector_config_opnsense, health_config):
    """Create a MetricsCollector instance with OPNsense enabled."""
    return MetricsCollector(collector_config_opnsense, health_config)


@pytest.mark.freebsd
class TestFreeBSDCollector:
    """Test collector functionality on FreeBSD."""

    def test_is_freebsd(self, mock_freebsd, collector):
        """Test FreeBSD platform detection."""
        assert collector._is_freebsd() is True

    def test_is_not_freebsd_on_linux(self, collector):
        """Test FreeBSD detection returns False on Linux."""
        with patch("platform.system", return_value="Linux"):
            assert collector._is_freebsd() is False

    @patch("subprocess.run")
    def test_collect_services_freebsd(self, mock_run, mock_freebsd, health_config, collector_config):
        """Test FreeBSD service status collection."""
        health = HealthConfig(services=["sshd", "nginx"], containers=[])
        collector = MetricsCollector(collector_config, health)

        # Mock service status responses
        def run_side_effect(cmd, **kwargs):
            if cmd[1] == "sshd":
                return Mock(returncode=0, stdout="sshd is running as pid 1234.", stderr="")
            elif cmd[1] == "nginx":
                return Mock(returncode=1, stdout="nginx is not running.", stderr="")
            return Mock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = run_side_effect

        services = collector._collect_services()

        assert len(services) == 2
        # sshd should be running
        sshd = next(s for s in services if s["name"] == "sshd")
        assert sshd["ok"] is True
        assert sshd["status"] == "active"
        # nginx should not be running
        nginx = next(s for s in services if s["name"] == "nginx")
        assert nginx["ok"] is False
        assert nginx["status"] == "inactive"

    @patch("subprocess.run")
    def test_collect_services_opnsense_pluginctl(self, mock_run, mock_freebsd, health_config, collector_config_opnsense):
        """Test OPNsense service status collection via pluginctl."""
        health = HealthConfig(services=["unbound", "tailscale", "webgui"], containers=[])
        collector = MetricsCollector(collector_config_opnsense, health)

        # Mock pluginctl status responses - return 0 for running services
        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "pluginctl":
                service = cmd[1]
                if service in ["unbound", "tailscale", "webgui"]:
                    # pluginctl returns 0 for running services
                    return Mock(returncode=0, stdout="", stderr="")
            return Mock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = run_side_effect

        services = collector._collect_services()

        assert len(services) == 3
        # All services should be running (pluginctl returned 0)
        for svc in services:
            assert svc["ok"] is True
            assert svc["status"] == "active"

    @patch("subprocess.run")
    def test_collect_services_opnsense_mixed(self, mock_run, mock_freebsd, health_config, collector_config_opnsense):
        """Test OPNsense service status with mix of running and stopped services."""
        health = HealthConfig(services=["unbound", "ids"], containers=[])
        collector = MetricsCollector(collector_config_opnsense, health)

        # Mock: unbound running via pluginctl, ids not running
        def run_side_effect(cmd, **kwargs):
            if cmd[0] == "pluginctl":
                service = cmd[1]
                if service == "unbound":
                    return Mock(returncode=0, stdout="", stderr="")
                elif service == "ids":
                    return Mock(returncode=1, stdout="", stderr="")
            return Mock(returncode=1, stdout="", stderr="")

        mock_run.side_effect = run_side_effect

        services = collector._collect_services()

        assert len(services) == 2
        unbound = next(s for s in services if s["name"] == "unbound")
        assert unbound["ok"] is True
        assert unbound["status"] == "active"

        ids = next(s for s in services if s["name"] == "ids")
        assert ids["ok"] is False
        assert ids["status"] == "inactive"

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_updates_freebsd(self, mock_run_command, mock_freebsd, collector):
        """Test FreeBSD package updates collection."""
        # Mock pkg version -vRL= output
        mock_run_command.side_effect = [
            # pkg version output - packages needing updates
            "pkg-1.19.0                         <   needs updating (remote has 1.20.0)\n"
            "python39-3.9.16                    <   needs updating (remote has 3.9.17)\n",
            # pkg audit output
            "0 problem(s) in the installed packages found.\n"
        ]

        updates = collector._collect_updates()

        assert updates is not None
        assert updates["source"] == "pkg"
        assert updates["pending"] == 2
        assert len(updates["items"]) == 2
        assert updates["items"][0]["name"] == "pkg-1.19.0"

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_updates_freebsd_with_security(self, mock_run_command, mock_freebsd, collector):
        """Test FreeBSD updates collection with security vulnerabilities."""
        mock_run_command.side_effect = [
            # pkg version output
            "",
            # pkg audit output with vulnerabilities
            "curl-8.0.0 is vulnerable:\n"
            "  CURL -- heap buffer overflow\n"
            "openssl-1.1.1 is vulnerable:\n"
            "  OpenSSL -- multiple vulnerabilities\n"
            "2 problem(s) in the installed packages found.\n"
        ]

        updates = collector._collect_updates()

        assert updates is not None
        assert updates["source"] == "pkg"
        assert updates["pending"] == 0
        assert updates["security_issues"] == 2

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_cpu_temps_freebsd(self, mock_run_command, mock_freebsd, collector):
        """Test FreeBSD CPU temperature collection via sysctl."""
        mock_run_command.return_value = (
            "dev.cpu.0.temperature: 45.0C\n"
            "dev.cpu.1.temperature: 47.0C\n"
            "dev.cpu.2.temperature: 46.0C\n"
            "dev.cpu.3.temperature: 48.0C\n"
        )

        temps = collector._collect_cpu_temps_freebsd()

        assert len(temps) == 4
        assert temps[0] == 45.0
        assert temps[1] == 47.0
        assert temps[2] == 46.0
        assert temps[3] == 48.0

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_cpu_temps_freebsd_kelvin(self, mock_run_command, mock_freebsd, collector):
        """Test FreeBSD CPU temperature conversion from Kelvin."""
        # Some systems report in Kelvin
        mock_run_command.return_value = (
            "dev.cpu.0.temperature: 318.15K\n"  # 45C in Kelvin
        )

        temps = collector._collect_cpu_temps_freebsd()

        assert len(temps) == 1
        assert abs(temps[0] - 45.0) < 0.1  # Should convert to ~45C

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_thermal_zones_freebsd(self, mock_run_command, mock_freebsd, collector):
        """Test FreeBSD thermal zone temperature collection."""
        mock_run_command.return_value = (
            "hw.acpi.thermal.tz0.temperature: 55.0C\n"
            "hw.acpi.thermal.tz1.temperature: 52.0C\n"
        )

        zones = collector._collect_thermal_zones_freebsd()

        assert len(zones) == 2
        assert zones[0]["name"] == "tz0"
        assert zones[0]["temp_c"] == 55.0
        assert zones[1]["name"] == "tz1"
        assert zones[1]["temp_c"] == 52.0


@pytest.mark.opnsense
class TestOPNsenseCollector:
    """Test OPNsense-specific collector functionality."""

    @patch("subprocess.run")
    def test_is_opnsense(self, mock_run, mock_freebsd, collector):
        """Test OPNsense detection."""
        mock_run.return_value = Mock(returncode=0, stdout="OPNsense 23.7", stderr="")
        assert collector._is_opnsense() is True

    @patch("subprocess.run")
    def test_is_not_opnsense(self, mock_run, mock_freebsd, collector):
        """Test OPNsense detection returns False on plain FreeBSD."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="command not found")
        assert collector._is_opnsense() is False

    def test_is_not_opnsense_on_linux(self, collector):
        """Test OPNsense detection returns False on non-FreeBSD."""
        with patch("platform.system", return_value="Linux"):
            assert collector._is_opnsense() is False

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_opnsense(self, mock_run_command, mock_freebsd, collector_opnsense):
        """Test OPNsense system information collection."""
        mock_run_command.side_effect = [
            "OPNsense 23.7.12_1",  # opnsense-version -v
            "System is running normally",  # configctl system status
        ]

        opnsense = collector_opnsense._collect_opnsense()

        assert opnsense is not None
        assert opnsense["version"] == "OPNsense 23.7.12_1"
        assert opnsense["status_raw"] == "System is running normally"

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_opnsense_disabled(self, mock_run_command, mock_freebsd, collector):
        """Test OPNsense collection when disabled."""
        opnsense = collector._collect_opnsense()
        assert opnsense is None
        mock_run_command.assert_not_called()

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_opnsense_plugins(self, mock_run_command, mock_freebsd, collector_opnsense):
        """Test OPNsense plugins collection."""
        mock_run_command.return_value = (
            "os-acme-client 3.16\n"
            "os-haproxy 4.0\n"
            "os-wireguard 2.1\n"
        )

        plugins = collector_opnsense._collect_opnsense_plugins()

        assert plugins is not None
        assert len(plugins) == 3
        assert plugins[0]["name"] == "os-acme-client"
        assert plugins[0]["version"] == "3.16"
        assert plugins[1]["name"] == "os-haproxy"
        assert plugins[2]["name"] == "os-wireguard"

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_opnsense_plugins_disabled(self, mock_run_command, mock_freebsd, collector):
        """Test OPNsense plugins collection when disabled."""
        plugins = collector._collect_opnsense_plugins()
        assert plugins is None
        mock_run_command.assert_not_called()

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_zenarmor_running(self, mock_run_command, mock_freebsd, collector_opnsense):
        """Test Zenarmor collection when services are running."""
        mock_run_command.side_effect = [
            "senpai is running as pid 1234",  # cloud status
            "eastpect is running as pid 5678",  # engine status
        ]

        zenarmor = collector_opnsense._collect_zenarmor()

        assert zenarmor is not None
        assert zenarmor["cloud_running"] is True
        assert zenarmor["cloud_pid"] == 1234
        assert zenarmor["engine_running"] is True
        assert zenarmor["engine_pid"] == 5678

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_zenarmor_not_running(self, mock_run_command, mock_freebsd, collector_opnsense):
        """Test Zenarmor collection when services are not running."""
        mock_run_command.side_effect = [
            "senpai is not running",  # cloud status
            "eastpect is not running",  # engine status
        ]

        zenarmor = collector_opnsense._collect_zenarmor()

        assert zenarmor is not None
        assert zenarmor["cloud_running"] is False
        assert "cloud_pid" not in zenarmor
        assert zenarmor["engine_running"] is False
        assert "engine_pid" not in zenarmor

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_zenarmor_disabled(self, mock_run_command, mock_freebsd, collector):
        """Test Zenarmor collection when disabled."""
        zenarmor = collector._collect_zenarmor()
        assert zenarmor is None
        mock_run_command.assert_not_called()

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_zenarmor_partial(self, mock_run_command, mock_freebsd, collector_opnsense):
        """Test Zenarmor collection when only one service is running."""
        mock_run_command.side_effect = [
            "senpai is running as pid 1234",  # cloud running
            None,  # engine command failed
        ]

        zenarmor = collector_opnsense._collect_zenarmor()

        assert zenarmor is not None
        assert zenarmor["cloud_running"] is True
        assert zenarmor["engine_running"] is False


@pytest.mark.freebsd
class TestFreeBSDIntegration:
    """Integration tests for FreeBSD payload generation."""

    def test_collect_returns_valid_payload_on_freebsd(self, mock_freebsd, collector):
        """Test that collect() returns a valid payload structure on FreeBSD."""
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
             patch("socket.gethostname", return_value="opnsense-fw"):

            payload = collector.collect()

            # Verify required fields
            assert "schema" in payload
            assert payload["schema"]["name"] == "hwmon-exporter"
            assert payload["schema"]["version"] == 1
            assert "ts" in payload
            assert "host" in payload
            assert payload["host"]["name"] == "opnsense-fw"
            assert "health" in payload
            assert "cpus" in payload
            assert "memory" in payload

    @patch("telemetry_tap.collector.MetricsCollector._run_command")
    def test_collect_includes_opnsense_sections(self, mock_run_command, mock_freebsd, collector_opnsense):
        """Test that collect() includes OPNsense sections when enabled."""
        def cpu_percent_side_effect(interval=None, percpu=False):
            if percpu:
                return [50.0]
            return 50.0

        # Set up mock responses for all the commands that will be called
        def run_command_side_effect(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "opnsense-version" in cmd_str:
                return "OPNsense 23.7"
            elif "configctl" in cmd_str:
                return "System running"
            elif "pkg" in cmd_str and "query" in cmd_str:
                return "os-wireguard 2.1\n"
            elif "zenarmorctl" in cmd_str and "cloud" in cmd_str:
                return "senpai is running as pid 1234"
            elif "zenarmorctl" in cmd_str and "engine" in cmd_str:
                return "eastpect is running as pid 5678"
            return None

        mock_run_command.side_effect = run_command_side_effect

        with patch("psutil.cpu_percent", side_effect=cpu_percent_side_effect), \
             patch("psutil.cpu_count", return_value=1), \
             patch("psutil.virtual_memory", return_value=Mock(used=1000, available=1000, percent=50.0, total=2000)), \
             patch("psutil.swap_memory", return_value=Mock(used=0, free=1000, percent=0.0)), \
             patch("psutil.boot_time", return_value=1000000), \
             patch("psutil.disk_partitions", return_value=[]), \
             patch("socket.gethostname", return_value="opnsense-fw"):

            payload = collector_opnsense.collect()

            # Verify OPNsense sections are included
            assert "opnsense" in payload
            assert payload["opnsense"]["version"] == "OPNsense 23.7"
            assert "opnsense_plugins" in payload
            assert len(payload["opnsense_plugins"]) == 1
            assert "zenarmor" in payload
            assert payload["zenarmor"]["cloud_running"] is True
            assert payload["zenarmor"]["engine_running"] is True
