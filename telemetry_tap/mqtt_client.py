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
        self._connected = False

        # Set up callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        if config.username:
            self.client.username_pw_set(config.username, config.password)
        if config.tls_enabled:
            self.client.tls_set(
                ca_certs=config.ca_cert,
                cert_reqs=ssl.CERT_REQUIRED,
            )

        # Set Last Will and Testament for availability
        self.client.will_set(
            self._availability_topic,
            payload="offline",
            qos=1,
            retain=True,
        )

        # Configure reconnect behavior with exponential backoff
        self.client.reconnect_delay_set(min_delay=1, max_delay=120)

    @property
    def _availability_topic(self) -> str:
        return f"{self.config.base_topic}/status"

    @property
    def connected(self) -> bool:
        return self._connected

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: dict[str, Any],
        rc: int,
    ) -> None:
        if rc == 0:
            self._connected = True
            self.logger.info(
                "Connected to MQTT broker %s:%s", self.config.host, self.config.port
            )
            # Publish online status
            self.client.publish(
                self._availability_topic,
                payload="online",
                qos=1,
                retain=True,
            )
        else:
            self._connected = False
            self.logger.error(
                "Failed to connect to MQTT broker, return code: %s", rc
            )

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        rc: int,
    ) -> None:
        self._connected = False
        if rc == 0:
            self.logger.info("Disconnected from MQTT broker (clean)")
        else:
            self.logger.warning(
                "Unexpectedly disconnected from MQTT broker, return code: %s. "
                "Will attempt to reconnect.",
                rc,
            )

    def connect(self) -> None:
        self.logger.info(
            "Connecting to MQTT broker %s:%s", self.config.host, self.config.port
        )
        self.client.connect(
            self.config.host,
            self.config.port,
            keepalive=self.config.keepalive,
        )
        # Start the network loop in the background for automatic reconnection
        self.client.loop_start()

    def disconnect(self) -> None:
        # Publish offline status before disconnecting
        if self._connected:
            self.client.publish(
                self._availability_topic,
                payload="offline",
                qos=1,
                retain=True,
            )
        self.client.loop_stop()
        self.client.disconnect()
        self.logger.info("Disconnected from MQTT broker")

    def publish_status(self, status: str) -> bool:
        """Publish a custom status to the availability topic.

        Args:
            status: Status string (e.g., "online", "offline", "sleeping")

        Returns:
            True if publish succeeded, False otherwise.
        """
        self.logger.info("Publishing status '%s' to %s", status, self._availability_topic)
        result = self.client.publish(
            self._availability_topic,
            payload=status,
            qos=1,
            retain=True,
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            self.logger.error("Failed to publish status, error code: %s", result.rc)
            return False
        return True

    def publish(self, payload: str) -> bool:
        if not self._connected:
            self.logger.warning(
                "Not connected to MQTT broker, message may be queued"
            )
        self.logger.debug("Publishing metrics payload to %s", self.config.base_topic)
        result = self.client.publish(
            self.config.base_topic,
            payload=payload,
            qos=self.config.qos,
            retain=self.config.retain,
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            self.logger.error("Failed to publish message, error code: %s", result.rc)
            return False
        return True

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
            "availability_topic": self._availability_topic,
            "payload_available": "online",
            "payload_not_available": "offline",
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
        # No longer needed when using loop_start(), but kept for compatibility
        # The background thread handles network events
        pass
