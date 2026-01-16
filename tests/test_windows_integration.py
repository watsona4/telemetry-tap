"""Integration tests for running telemetry-tap on Windows."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from telemetry_tap.main import main, build_parser
from telemetry_tap.config import load_config


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_content = """[mqtt]
host = localhost
port = 1883
base_topic = telemetry/hwmon
discovery_topic = homeassistant
client_id = telemetry-tap-test
username =
password =
qos = 0
retain = false

[publish]
interval_s = 15

[collector]
smartctl_path = smartctl
lsblk_path = lsblk
sensors_path = sensors
dmidecode_path = dmidecode
librehardwaremonitor_url = http://localhost:8085/data.json

[health]
services =
containers =
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def mock_windows_system():
    """Mock Windows platform and common psutil calls."""
    with patch("platform.system", return_value="Windows"), \
         patch("platform.platform", return_value="Windows-10-10.0.19045-SP0"), \
         patch("platform.release", return_value="10"), \
         patch("platform.machine", return_value="AMD64"), \
         patch("platform.python_version", return_value="3.10.0"), \
         patch("socket.gethostname", return_value="TEST-PC"), \
         patch("psutil.boot_time", return_value=1000000.0), \
         patch("psutil.cpu_percent") as mock_cpu_pct, \
         patch("psutil.cpu_count", return_value=8), \
         patch("psutil.virtual_memory") as mock_vm, \
         patch("psutil.swap_memory") as mock_swap, \
         patch("psutil.disk_partitions") as mock_parts, \
         patch("psutil.disk_io_counters", return_value={}), \
         patch("psutil.net_if_addrs", return_value={}), \
         patch("psutil.net_io_counters", return_value={}), \
         patch("psutil.net_if_stats", return_value={}):

        # Setup default mock return values
        mock_cpu_pct.side_effect = lambda interval=None, percpu=False: (
            [25.0] * 8 if percpu else 25.0
        )
        mock_vm.return_value = Mock(
            used=8589934592,
            available=8589934592,
            percent=50.0,
            total=17179869184,
        )
        mock_swap.return_value = Mock(
            used=0,
            free=1073741824,
            percent=0.0,
        )
        mock_parts.return_value = [
            Mock(
                device="C:\\",
                mountpoint="C:\\",
                fstype="NTFS",
            )
        ]

        yield


@pytest.mark.windows
@pytest.mark.integration
class TestWindowsIntegration:
    """Integration tests for Windows platform."""

    def test_config_loading(self, temp_config_file):
        """Test that config loads properly."""
        config = load_config(temp_config_file)

        assert config.mqtt.host == "localhost"
        assert config.mqtt.port == 1883
        assert config.publish.interval_s == 15
        assert config.collector.librehardwaremonitor_url == "http://localhost:8085/data.json"

    def test_argument_parser(self):
        """Test command-line argument parser."""
        parser = build_parser()

        # Test default arguments
        args = parser.parse_args([])
        assert args.config == "config/example.cfg"
        assert args.log_level == "INFO"
        assert args.verbose == 0
        assert args.dry_run is False
        assert args.once is False

        # Test custom arguments
        args = parser.parse_args([
            "--config", "test.cfg",
            "--log-level", "DEBUG",
            "-vv",
            "--dry-run",
            "--once",
            "--dump-json", "output.json",
        ])
        assert args.config == "test.cfg"
        assert args.log_level == "DEBUG"
        assert args.verbose == 2
        assert args.dry_run is True
        assert args.once is True
        assert args.dump_json == "output.json"

    @patch("telemetry_tap.main.MqttPublisher")
    @patch("psutil.disk_usage")
    def test_dry_run_mode_on_windows(
        self,
        mock_disk_usage,
        mock_publisher,
        temp_config_file,
        mock_windows_system,
    ):
        """Test that dry-run mode works on Windows without MQTT."""
        mock_disk_usage.return_value = Mock(
            used=107374182400,
            free=107374182400,
            total=214748364800,
        )
        mock_mqtt_instance = Mock()
        mock_publisher.return_value = mock_mqtt_instance

        with patch("sys.argv", ["telemetry-tap", "--config", temp_config_file, "--dry-run", "--once"]):
            try:
                main()
            except SystemExit:
                pass

        # In dry-run mode, MQTT is instantiated but connect() is not called
        mock_mqtt_instance.connect.assert_not_called()

    @patch("telemetry_tap.main.MqttPublisher")
    @patch("psutil.disk_usage")
    def test_once_mode_on_windows(
        self,
        mock_disk_usage,
        mock_publisher,
        temp_config_file,
        mock_windows_system,
    ):
        """Test that --once mode collects single payload and exits."""
        mock_disk_usage.return_value = Mock(
            used=107374182400,
            free=107374182400,
            total=214748364800,
        )
        mock_mqtt_instance = Mock()
        mock_publisher.return_value = mock_mqtt_instance

        with patch("sys.argv", ["telemetry-tap", "--config", temp_config_file, "--once", "--dry-run"]):
            try:
                main()
            except SystemExit:
                pass

        # Should not enter the infinite loop
        mock_mqtt_instance.loop.assert_not_called()

    @patch("telemetry_tap.main.MqttPublisher")
    @patch("psutil.disk_usage")
    def test_json_dump_on_windows(
        self,
        mock_disk_usage,
        mock_publisher,
        temp_config_file,
        mock_windows_system,
    ):
        """Test that --dump-json creates a valid JSON file on Windows."""
        mock_disk_usage.return_value = Mock(
            used=107374182400,
            free=107374182400,
            total=214748364800,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            with patch("sys.argv", [
                "telemetry-tap",
                "--config", temp_config_file,
                "--dry-run",
                "--once",
                "--dump-json", output_file,
            ]):
                try:
                    main()
                except SystemExit:
                    pass

            # Verify JSON file was created and is valid
            assert Path(output_file).exists()
            with open(output_file, "r") as f:
                payload = json.load(f)

            # Verify payload structure
            assert "schema" in payload
            assert payload["schema"]["name"] == "hwmon-exporter"
            assert "host" in payload
            assert payload["host"]["name"] == "TEST-PC"
            assert payload["host"]["system"] == "Windows"

        finally:
            # Cleanup
            Path(output_file).unlink(missing_ok=True)

    @patch("telemetry_tap.collector.urlopen")
    @patch("telemetry_tap.main.MqttPublisher")
    @patch("psutil.disk_usage")
    def test_full_collection_with_lhm_on_windows(
        self,
        mock_disk_usage,
        mock_publisher,
        mock_urlopen,
        temp_config_file,
        mock_windows_system,
    ):
        """Test full collection cycle with LibreHardwareMonitor on Windows."""
        mock_disk_usage.return_value = Mock(
            used=107374182400,
            free=107374182400,
            total=214748364800,
        )

        # Mock LHM response
        lhm_data = {
            "Children": [
                {
                    "Type": "Hardware",
                    "Text": "AMD Ryzen 9 5900X",
                    "HardwareType": "Cpu",
                    "Children": [],
                },
                {
                    "Type": "Hardware",
                    "Text": "NVIDIA GeForce RTX 3080",
                    "HardwareType": "GpuNvidia",
                    "Children": [
                        {
                            "Type": "Sensor",
                            "Text": "GPU Core",
                            "SensorType": "load",
                            "Value": 50.0,
                        }
                    ],
                },
            ]
        }
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(lhm_data).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            with patch("sys.argv", [
                "telemetry-tap",
                "--config", temp_config_file,
                "--dry-run",
                "--once",
                "--dump-json", output_file,
            ]):
                try:
                    main()
                except SystemExit:
                    pass

            # Verify payload includes LHM data
            with open(output_file, "r") as f:
                payload = json.load(f)

            # Should have GPU data from LHM
            assert "gpus" in payload
            assert len(payload["gpus"]) == 1
            assert payload["gpus"][0]["name"] == "NVIDIA GeForce RTX 3080"

        finally:
            Path(output_file).unlink(missing_ok=True)

    @patch("telemetry_tap.main.validate_payload")
    @patch("telemetry_tap.main.MqttPublisher")
    @patch("psutil.disk_usage")
    def test_schema_validation_on_windows(
        self,
        mock_disk_usage,
        mock_publisher,
        mock_validate,
        temp_config_file,
        mock_windows_system,
    ):
        """Test that schema validation runs on Windows."""
        mock_disk_usage.return_value = Mock(
            used=107374182400,
            free=107374182400,
            total=214748364800,
        )
        mock_validate.return_value = []  # No errors

        with patch("sys.argv", ["telemetry-tap", "--config", temp_config_file, "--dry-run", "--once"]):
            try:
                main()
            except SystemExit:
                pass

        # Validate should have been called at least once
        assert mock_validate.call_count >= 1

    @patch("psutil.disk_usage")
    def test_windows_specific_filesystem_paths(
        self,
        mock_disk_usage,
        temp_config_file,
        mock_windows_system,
    ):
        """Test that Windows filesystem paths are handled correctly."""
        mock_disk_usage.return_value = Mock(
            used=107374182400,
            free=107374182400,
            total=214748364800,
        )

        from telemetry_tap.collector import MetricsCollector
        from telemetry_tap.config import load_config

        config = load_config(temp_config_file)
        collector = MetricsCollector(config.collector, config.health)

        with patch("psutil.disk_partitions") as mock_parts:
            mock_parts.return_value = [
                Mock(device="C:\\", mountpoint="C:\\", fstype="NTFS"),
                Mock(device="D:\\", mountpoint="D:\\", fstype="NTFS"),
            ]

            filesystems = collector._collect_filesystems()

            assert len(filesystems) == 2
            assert filesystems[0]["name"] == "C:\\"
            assert filesystems[0]["mountpoint"] == "C:\\"
            assert filesystems[1]["name"] == "D:\\"
            assert filesystems[1]["mountpoint"] == "D:\\"


@pytest.mark.windows
@pytest.mark.integration
def test_can_import_all_modules():
    """Test that all modules can be imported on Windows."""
    import telemetry_tap.main
    import telemetry_tap.collector
    import telemetry_tap.config
    import telemetry_tap.mqtt_client
    import telemetry_tap.schema
    import telemetry_tap.logging_utils

    assert telemetry_tap.main is not None
    assert telemetry_tap.collector is not None
    assert telemetry_tap.config is not None
    assert telemetry_tap.mqtt_client is not None
    assert telemetry_tap.schema is not None
    assert telemetry_tap.logging_utils is not None
