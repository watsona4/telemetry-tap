from __future__ import annotations

import json
import logging
import ssl
from typing import Any

import paho.mqtt.client as mqtt

from telemetry_tap.config import MqttConfig


class MqttPublisher:
    def __init__(self, config: MqttConfig) -> None:
        self.config = config
        self.client = mqtt.Client(client_id=config.client_id, protocol=mqtt.MQTTv311)
        self.logger = logging.getLogger(self.__class__.__name__)
        if config.username:
            self.client.username_pw_set(config.username, config.password)
        if config.tls_enabled:
            self.client.tls_set(
                ca_certs=config.ca_cert,
                cert_reqs=ssl.CERT_REQUIRED,
            )

    def connect(self) -> None:
        self.logger.info("Connecting to MQTT broker %s:%s", self.config.host, self.config.port)
        self.client.connect(self.config.host, self.config.port)

    def publish(self, payload: str) -> None:
        self.logger.debug("Publishing metrics payload to %s", self.config.base_topic)
        self.client.publish(
            self.config.base_topic,
            payload=payload,
            qos=self.config.qos,
            retain=self.config.retain,
        )

    def publish_discovery(self, host_payload: dict[str, Any]) -> None:
        host = host_payload.get("host", {})
        device_id = self.config.client_id
        name = host.get("name", device_id)
        discovery_payload = {
            "name": f"{name} Hardware Telemetry",
            "unique_id": f"{device_id}_telemetry",
            "state_topic": self.config.base_topic,
            "value_template": "{{ value_json.host.uptime_s }}",
            "json_attributes_topic": self.config.base_topic,
            "device": {
                "identifiers": [device_id],
                "name": name,
                "model": host.get("machine"),
                "manufacturer": host.get("system"),
            },
        }
        topic = (
            f"{self.config.discovery_topic}/sensor/{device_id}/telemetry/config"
        )
        self.logger.debug("Publishing Home Assistant discovery to %s", topic)
        self.client.publish(
            topic,
            payload=json.dumps(discovery_payload),
            qos=self.config.qos,
            retain=True,
        )

    def loop(self) -> None:
        self.client.loop(timeout=1.0)
