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
    librehardwaremonitor_url: str | None


@dataclass(frozen=True)
class AppConfig:
    mqtt: MqttConfig
    publish: PublishConfig
    collector: CollectorConfig


def _get_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def load_config(path: str | Path) -> AppConfig:
    parser = configparser.ConfigParser()
    read_files = parser.read(path)
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {path}")

    mqtt_section = parser["mqtt"]
    publish_section = parser["publish"]
    collector_section = parser["collector"] if "collector" in parser else {}

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

    collector = CollectorConfig(
        smartctl_path=str(collector_section.get("smartctl_path", "smartctl")),
        lsblk_path=str(collector_section.get("lsblk_path", "lsblk")),
        sensors_path=str(collector_section.get("sensors_path", "sensors")),
        dmidecode_path=str(collector_section.get("dmidecode_path", "dmidecode")),
        librehardwaremonitor_url=_get_optional(
            collector_section.get("librehardwaremonitor_url")
        ),
    )

    return AppConfig(mqtt=mqtt, publish=publish, collector=collector)
