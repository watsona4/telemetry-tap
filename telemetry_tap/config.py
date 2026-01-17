from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import configparser


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int
    base_topic: str
    discovery_topic: str
    client_id: str
    username: str | None
    password: str | None
    qos: int
    retain: bool
    tls_enabled: bool
    ca_cert: str | None


@dataclass(frozen=True)
class PublishConfig:
    interval_s: int


@dataclass(frozen=True)
class CollectorConfig:
    smartctl_path: str
    lsblk_path: str
    sensors_path: str
    dmidecode_path: str
    apt_path: str
    dnf_path: str
    systemctl_path: str
    librehardwaremonitor_url: str | None
    intel_gpu_top_path: str
    borg_path: str
    borg_repos: list[str]
    enable_tpu: bool
    # Time server configuration
    enable_time_server: bool
    chronyc_path: str
    gpspipe_path: str
    pps_device: str | None
    # FreeBSD tool paths
    sysctl_path: str
    pkg_path: str
    service_path: str
    # OPNsense-specific tool paths
    pluginctl_path: str
    configctl_path: str
    zenarmorctl_path: str
    # Feature toggles
    enable_opnsense: bool
    enable_zenarmor: bool


@dataclass(frozen=True)
class HealthConfig:
    services: list[str]
    containers: list[str]


@dataclass(frozen=True)
class AppConfig:
    mqtt: MqttConfig
    publish: PublishConfig
    collector: CollectorConfig
    health: HealthConfig


def _get_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _get_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config(path: str | Path) -> AppConfig:
    parser = configparser.ConfigParser()
    read_files = parser.read(path)
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {path}")

    mqtt_section = parser["mqtt"]
    publish_section = parser["publish"]

    mqtt = MqttConfig(
        host=mqtt_section.get("host", "localhost"),
        port=mqtt_section.getint("port", 1883),
        base_topic=mqtt_section.get("base_topic", "telemetry/hwmon"),
        discovery_topic=mqtt_section.get("discovery_topic", "homeassistant"),
        client_id=mqtt_section.get("client_id", "telemetry-tap"),
        username=_get_optional(mqtt_section.get("username")),
        password=_get_optional(mqtt_section.get("password")),
        qos=mqtt_section.getint("qos", 0),
        retain=mqtt_section.getboolean("retain", False),
        tls_enabled=mqtt_section.getboolean("tls", False),
        ca_cert=_get_optional(mqtt_section.get("ca_cert")),
    )

    publish = PublishConfig(
        interval_s=publish_section.getint("interval_s", 15),
    )

    # Use parser.get/getboolean with fallback to handle missing [collector] section
    collector = CollectorConfig(
        smartctl_path=parser.get("collector", "smartctl_path", fallback="smartctl"),
        lsblk_path=parser.get("collector", "lsblk_path", fallback="lsblk"),
        sensors_path=parser.get("collector", "sensors_path", fallback="sensors"),
        dmidecode_path=parser.get("collector", "dmidecode_path", fallback="dmidecode"),
        apt_path=parser.get("collector", "apt_path", fallback="apt"),
        dnf_path=parser.get("collector", "dnf_path", fallback="dnf"),
        systemctl_path=parser.get("collector", "systemctl_path", fallback="systemctl"),
        librehardwaremonitor_url=_get_optional(
            parser.get("collector", "librehardwaremonitor_url", fallback=None)
        ),
        intel_gpu_top_path=parser.get("collector", "intel_gpu_top_path", fallback="intel_gpu_top"),
        borg_path=parser.get("collector", "borg_path", fallback="borg"),
        borg_repos=_get_list(parser.get("collector", "borg_repos", fallback=None)),
        enable_tpu=parser.getboolean("collector", "enable_tpu", fallback=True),
        enable_time_server=parser.getboolean("collector", "enable_time_server", fallback=False),
        chronyc_path=parser.get("collector", "chronyc_path", fallback="chronyc"),
        gpspipe_path=parser.get("collector", "gpspipe_path", fallback="gpspipe"),
        pps_device=_get_optional(parser.get("collector", "pps_device", fallback=None)),
        # FreeBSD tool paths
        sysctl_path=parser.get("collector", "sysctl_path", fallback="sysctl"),
        pkg_path=parser.get("collector", "pkg_path", fallback="pkg"),
        service_path=parser.get("collector", "service_path", fallback="service"),
        # OPNsense-specific tool paths
        pluginctl_path=parser.get("collector", "pluginctl_path", fallback="pluginctl"),
        configctl_path=parser.get("collector", "configctl_path", fallback="configctl"),
        zenarmorctl_path=parser.get("collector", "zenarmorctl_path", fallback="zenarmorctl"),
        # Feature toggles
        enable_opnsense=parser.getboolean("collector", "enable_opnsense", fallback=False),
        enable_zenarmor=parser.getboolean("collector", "enable_zenarmor", fallback=False),
    )

    # Use parser.get with fallback to handle missing [health] section
    health = HealthConfig(
        services=_get_list(parser.get("health", "services", fallback=None)),
        containers=_get_list(parser.get("health", "containers", fallback=None)),
    )

    return AppConfig(mqtt=mqtt, publish=publish, collector=collector, health=health)
