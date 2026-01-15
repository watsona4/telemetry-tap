from __future__ import annotations

import argparse
import json
import logging
import time

from telemetry_tap.collector import MetricsCollector
from telemetry_tap.config import load_config
from telemetry_tap.mqtt_client import MqttPublisher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemetry Tap hardware exporter")
    parser.add_argument(
        "--config",
        default="config/example.cfg",
        help="Path to CFG configuration file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log payloads without publishing to MQTT",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(message)s")
    logger = logging.getLogger("telemetry_tap")
    config = load_config(args.config)

    collector = MetricsCollector(config.collector)
    publisher = MqttPublisher(config.mqtt)
    if not args.dry_run:
        publisher.connect()

    initial_payload = collector.collect()
    initial_json = json.dumps(initial_payload)
    if args.dry_run:
        logger.info("Dry run enabled; skipping MQTT publish.")
        logger.debug("Initial payload: %s", initial_json)
    else:
        publisher.publish_discovery(initial_payload)
        publisher.publish(initial_json)
        publisher.loop()

    interval = max(1, config.publish.interval_s)
    logging.info("Telemetry Tap started. Publishing every %s seconds.", interval)

    try:
        while True:
            payload = collector.collect()
            payload_json = json.dumps(payload)
            if args.dry_run:
                logger.debug("Payload: %s", payload_json)
            else:
                publisher.publish(payload_json)
                publisher.loop()
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Telemetry Tap stopped.")


if __name__ == "__main__":
    main()
