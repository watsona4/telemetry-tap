"""Tests for time server metrics collection (chrony, gpsd, PPS)."""
from __future__ import annotations

from unittest.mock import Mock, patch
import pytest

from telemetry_tap.collector import MetricsCollector
from telemetry_tap.config import CollectorConfig, HealthConfig


@pytest.fixture
def time_server_config():
    """Create a collector config with time server enabled."""
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
        enable_time_server=True,
        chronyc_path="chronyc",
        gpspipe_path="gpspipe",
        pps_device="/dev/pps0",
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
def health_config():
    """Create a health config."""
    return HealthConfig(services=[], containers=[], exclude_disk_checks=[], enable_hassio=False, ha_path="ha")


@pytest.fixture
def collector(time_server_config, health_config):
    """Create a MetricsCollector instance with time server enabled."""
    return MetricsCollector(time_server_config, health_config)


class TestChronyTracking:
    """Tests for chrony tracking collection."""

    def test_parse_tracking_output(self, collector):
        """Test parsing chronyc tracking output."""
        tracking_output = """Reference ID    : E1FE1EBE (time.cloudflare.com)
Stratum         : 4
Ref time (UTC)  : Fri Jan 16 23:43:54 2026
System time     : 0.000226142 seconds slow of NTP time
Last offset     : -0.000308556 seconds
RMS offset      : 0.000309569 seconds
Frequency       : 14.770 ppm fast
Residual freq   : -0.006 ppm
Skew            : 0.094 ppm
Root delay      : 0.026936660 seconds
Root dispersion : 0.001603267 seconds
Update interval : 1036.9 seconds
Leap status     : Normal"""

        with patch.object(collector, "_run_command", return_value=tracking_output):
            result = collector._collect_chrony_tracking()

        assert result is not None
        assert result["reference_id"] == "E1FE1EBE"
        assert result["reference_name"] == "time.cloudflare.com"
        assert result["stratum"] == 4
        assert result["system_time_offset_s"] == pytest.approx(-0.000226142, rel=1e-6)
        assert result["last_offset_s"] == pytest.approx(-0.000308556, rel=1e-6)
        assert result["rms_offset_s"] == pytest.approx(0.000309569, rel=1e-6)
        assert result["frequency_ppm"] == pytest.approx(14.770, rel=1e-3)
        assert result["residual_freq_ppm"] == pytest.approx(-0.006, rel=1e-3)
        assert result["skew_ppm"] == pytest.approx(0.094, rel=1e-3)
        assert result["root_delay_s"] == pytest.approx(0.026936660, rel=1e-6)
        assert result["root_dispersion_s"] == pytest.approx(0.001603267, rel=1e-6)
        assert result["update_interval_s"] == pytest.approx(1036.9, rel=1e-1)
        assert result["leap_status"] == "Normal"

    def test_tracking_not_available(self, collector):
        """Test when chronyc is not available."""
        with patch.object(collector, "_run_command", return_value=None):
            result = collector._collect_chrony_tracking()

        assert result is None


class TestChronySources:
    """Tests for chrony sources collection."""

    def test_parse_sources_output(self, collector):
        """Test parsing chronyc sources output."""
        sources_output = """MS Name/IP address         Stratum Poll Reach LastRx Last sample
===============================================================================
#- NMEA                          0   4   377    22    +21ms[  +21ms] +/-  100ms
#? PPS                           0   4     0    6d  +7028us[+1773ns] +/-  737ns
^* time.cloudflare.com           3  10   377   501  -1913us[-2222us] +/-   15ms
^- sth1-ts.nts.netnod.se         1  10   377   840  -2658us[-2961us] +/-   52ms"""

        stats_output = """Name/IP Address            NP  NR  Span  Frequency  Freq Skew  Offset  Std Dev
==============================================================================
NMEA                       30  15   466     -0.425      0.766    +21ms   167us
PPS                        11   6   158     +0.018      0.018    +17ms   451ns
time.cloudflare.com        23  13  414m     -0.006      0.085  -3166ns   758us
sth1-ts.nts.netnod.se      18  11  327m     -0.014      0.099   +352us   669us"""

        def mock_run_command(cmd):
            if "sources" in cmd:
                return sources_output
            elif "sourcestats" in cmd:
                return stats_output
            return None

        with patch.object(collector, "_run_command", side_effect=mock_run_command):
            result = collector._collect_chrony_sources()

        assert result is not None
        assert len(result) == 4

        # Check NMEA source
        nmea = result[0]
        assert nmea["name"] == "NMEA"
        assert nmea["mode"] == "local"
        assert nmea["state"] == "not_combined"
        assert nmea["stratum"] == 0
        assert nmea["poll_interval"] == 4
        assert nmea["reachability"] == 0o377
        assert nmea["samples"] == 30

        # Check PPS source
        pps = result[1]
        assert pps["name"] == "PPS"
        assert pps["mode"] == "local"
        assert pps["state"] == "unusable"

        # Check NTP server source
        cloudflare = result[2]
        assert cloudflare["name"] == "time.cloudflare.com"
        assert cloudflare["mode"] == "server"
        assert cloudflare["state"] == "sync"
        assert cloudflare["stratum"] == 3

    def test_sources_not_available(self, collector):
        """Test when chronyc sources is not available."""
        with patch.object(collector, "_run_command", return_value=None):
            result = collector._collect_chrony_sources()

        assert result is None


class TestChronyServerStats:
    """Tests for chrony server statistics collection."""

    def test_parse_serverstats_output(self, collector):
        """Test parsing chronyc serverstats output."""
        serverstats_output = """NTP packets received       : 293064
NTP packets dropped        : 0
Command packets received   : 1065047
Command packets dropped    : 0
Client log records dropped : 0
NTS-KE connections accepted: 0
NTS-KE connections dropped : 0
Authenticated NTP packets  : 0
Interleaved NTP packets    : 0"""

        clients_output = """Hostname                      NTP   Drop Int IntL Last     Cmd   Drop Int  Last
===============================================================================
10.10.30.106                  510      0  12   -   33m       0      0   -     -
back-porch.home.arpa         1022      0  11   -   260       0      0   -     -
side-yard-camera.home.ar>    3049      0   9   -   137       0      0   -     -"""

        def mock_run_command(cmd):
            if "serverstats" in cmd:
                return serverstats_output
            elif "clients" in cmd:
                return clients_output
            return None

        with patch.object(collector, "_run_command", side_effect=mock_run_command):
            result = collector._collect_chrony_serverstats()

        assert result is not None
        assert result["ntp_packets_received"] == 293064
        assert result["ntp_packets_dropped"] == 0
        assert result["cmd_packets_received"] == 1065047
        assert result["cmd_packets_dropped"] == 0
        assert result["nts_ke_connections"] == 0
        assert result["authenticated_packets"] == 0
        assert result["interleaved_packets"] == 0
        assert result["client_count"] == 3


class TestGpsd:
    """Tests for gpsd data collection."""

    def test_parse_gpspipe_output(self, collector):
        """Test parsing gpspipe JSON output."""
        gpspipe_output = """{"class":"VERSION","release":"3.22","rev":"3.22","proto_major":3,"proto_minor":14}
{"class":"DEVICES","devices":[{"class":"DEVICE","path":"/dev/ttyACM0","driver":"u-blox","subtype":"SW EXT CORE 1.00","subtype1":"ROM BASE,FWVER=HPG 1.32,MOD=ZED-F9P,GPS;GLO;GAL;BDS","activated":"2026-01-16T23:53:12.024Z"}]}
{"class":"TPV","device":"/dev/ttyACM0","status":2,"mode":3,"time":"2026-01-16T23:53:13.000Z","lat":40.7128,"lon":-74.0060,"alt":10.5,"speed":0.0,"leapseconds":18,"ept":0.005}
{"class":"SKY","device":"/dev/ttyACM0","hdop":1.2,"vdop":1.5,"pdop":1.9,"tdop":0.8,"nSat":12,"uSat":8,"satellites":[{"PRN":3,"el":15.0,"az":39.0,"ss":25.0,"used":true,"gnssid":0,"health":1},{"PRN":6,"el":77.0,"az":352.0,"ss":35.0,"used":true,"gnssid":0,"health":1}]}"""

        def mock_run_command(cmd, stderr=None):
            return gpspipe_output

        with patch.object(collector, "_run_command", side_effect=mock_run_command):
            result = collector._collect_gpsd()

        assert result is not None
        assert result["device"] == "/dev/ttyACM0"
        assert result["driver"] == "u-blox"
        assert result["mode"] == 3
        assert result["status"] == 2
        assert result["latitude"] == pytest.approx(40.7128, rel=1e-4)
        assert result["longitude"] == pytest.approx(-74.0060, rel=1e-4)
        assert result["altitude_m"] == pytest.approx(10.5, rel=1e-1)
        assert result["leapseconds"] == 18
        assert result["hdop"] == pytest.approx(1.2, rel=1e-1)
        assert result["satellites_visible"] == 12
        assert result["satellites_used"] == 8
        assert len(result["satellites"]) == 2
        assert result["satellites"][0]["prn"] == 3
        assert result["satellites"][0]["used"] is True

    def test_gpsd_not_available(self, collector):
        """Test when gpspipe is not available."""
        with patch.object(collector, "_run_command", return_value=None):
            result = collector._collect_gpsd()

        assert result is None


class TestPps:
    """Tests for PPS data collection."""

    def test_parse_pps_sysfs(self, collector):
        """Test parsing PPS sysfs data."""
        assert_data = "1768068085.000001313#1291306"
        clear_data = "1768068085.500000123#1291306"

        def mock_read_file(path):
            if "assert" in path:
                return assert_data
            elif "clear" in path:
                return clear_data
            return None

        with patch.object(collector, "_read_file", side_effect=mock_read_file):
            result = collector._collect_pps()

        assert result is not None
        assert result["device"] == "/dev/pps0"
        assert result["assert_timestamp_s"] == pytest.approx(1768068085.000001313, rel=1e-9)
        assert result["assert_sequence"] == 1291306
        assert result["clear_timestamp_s"] == pytest.approx(1768068085.500000123, rel=1e-9)
        assert result["clear_sequence"] == 1291306

    def test_pps_not_available(self, collector):
        """Test when PPS sysfs is not available."""
        with patch.object(collector, "_read_file", return_value=None):
            result = collector._collect_pps()

        assert result is None


class TestTimeServices:
    """Tests for time service status collection."""

    def test_all_services_active(self, collector):
        """Test when all time services are active."""
        def mock_run_command(cmd):
            if "is-active" in cmd:
                return "active"
            return None

        with patch.object(collector, "_run_command", side_effect=mock_run_command):
            result = collector._collect_time_services()

        assert result is not None
        assert result["chronyd"] is True
        assert result["gpsd"] is True
        assert result["ntrip"] is True

    def test_mixed_service_states(self, collector):
        """Test with mixed service states."""
        def mock_run_command(cmd):
            if "chronyd" in cmd:
                return "active"
            elif "gpsd" in cmd:
                return "active"
            elif "str2str" in cmd:
                return "inactive"
            return None

        with patch.object(collector, "_run_command", side_effect=mock_run_command):
            result = collector._collect_time_services()

        assert result is not None
        assert result["chronyd"] is True
        assert result["gpsd"] is True
        assert result["ntrip"] is False


class TestTimeValueParsing:
    """Tests for time value parsing utilities."""

    def test_parse_time_value_nanoseconds(self, collector):
        """Test parsing nanosecond values."""
        assert collector._parse_time_value("+1773ns") == pytest.approx(1.773e-6, rel=1e-9)
        assert collector._parse_time_value("-500ns") == pytest.approx(-5e-7, rel=1e-9)

    def test_parse_time_value_microseconds(self, collector):
        """Test parsing microsecond values."""
        assert collector._parse_time_value("+7028us") == pytest.approx(0.007028, rel=1e-6)
        assert collector._parse_time_value("-2222us") == pytest.approx(-0.002222, rel=1e-6)

    def test_parse_time_value_milliseconds(self, collector):
        """Test parsing millisecond values."""
        assert collector._parse_time_value("+21ms") == pytest.approx(0.021, rel=1e-3)
        assert collector._parse_time_value("-15ms") == pytest.approx(-0.015, rel=1e-3)

    def test_parse_time_interval_days(self, collector):
        """Test parsing day intervals."""
        assert collector._parse_time_interval("6d") == pytest.approx(518400, rel=1)

    def test_parse_time_interval_hours(self, collector):
        """Test parsing hour intervals."""
        assert collector._parse_time_interval("29h") == pytest.approx(104400, rel=1)

    def test_parse_time_interval_minutes(self, collector):
        """Test parsing minute intervals."""
        assert collector._parse_time_interval("33m") == pytest.approx(1980, rel=1)

    def test_parse_time_interval_seconds(self, collector):
        """Test parsing plain second values."""
        assert collector._parse_time_interval("501") == pytest.approx(501, rel=1)
        assert collector._parse_time_interval("-") == 0.0


class TestTimeServerIntegration:
    """Integration tests for full time server collection."""

    def test_full_time_server_collection(self, collector):
        """Test full time server metrics collection."""
        tracking_output = """Reference ID    : E1FE1EBE (time.cloudflare.com)
Stratum         : 1
Leap status     : Normal"""

        sources_output = """MS Name/IP address         Stratum Poll Reach LastRx Last sample
===============================================================================
#* PPS                           0   4   377    1  +100ns[+100ns] +/-  500ns"""

        gpspipe_output = """{"class":"TPV","mode":3,"lat":40.7128,"lon":-74.0060}"""

        def mock_run_command(cmd, stderr=None):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "tracking" in cmd_str:
                return tracking_output
            elif "sources" in cmd_str:
                return sources_output
            elif "sourcestats" in cmd_str:
                return ""
            elif "serverstats" in cmd_str:
                return None
            elif "gpspipe" in cmd_str:
                return gpspipe_output
            elif "is-active" in cmd_str:
                return "active"
            return None

        def mock_read_file(path):
            if "assert" in path:
                return "1768068085.000001313#1291306"
            return None

        with patch.object(collector, "_run_command", side_effect=mock_run_command), \
             patch.object(collector, "_read_file", side_effect=mock_read_file):
            result = collector._collect_time_server()

        assert result is not None
        assert "tracking" in result
        assert result["tracking"]["stratum"] == 1
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert "gps" in result
        assert result["gps"]["latitude"] == pytest.approx(40.7128, rel=1e-4)
        assert "pps" in result
        assert "services" in result
