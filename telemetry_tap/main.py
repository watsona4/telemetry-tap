from __future__ import annotations

import argparse
import json
import logging
import time

from telemetry_tap.collector import MetricsCollector
from telemetry_tap.config import load_config
from telemetry_tap.logging_utils import configure_logging, resolve_log_level
from telemetry_tap.mqtt_client import MqttPublisher
from telemetry_tap.schema import validate_payload


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
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable debug logging (-v) or trace logging (-vv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log payloads without publishing to MQTT",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect and publish a single payload, then exit",
    )
    parser.add_argument(
        "--dump-json",
        help="Write the JSON payload to a file (overwrites on each loop)",
    )
    parser.add_argument(
        "--publish-status",
        metavar="STATUS",
        help="Publish a status (e.g., 'sleeping', 'online') to the availability topic and exit. "
             "Useful for system sleep/wake hooks.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    level = resolve_log_level(args.verbose, args.log_level)
    configure_logging(level)
    logger = logging.getLogger("telemetry_tap")
    config = load_config(args.config)
    pretty_print = level <= logging.DEBUG

    # Handle --publish-status mode (quick publish and exit)
    if args.publish_status:
        publisher = MqttPublisher(config.mqtt)
        publisher.connect()
        # Wait briefly for connection to establish
        time.sleep(0.5)
        if publisher.connected:
            publisher.publish_status(args.publish_status)
            # Wait for message delivery
            time.sleep(0.5)
        else:
            logger.error("Failed to connect to MQTT broker")
        publisher.disconnect()
        return

    collector = MetricsCollector(config.collector, config.health)
    publisher = None if args.dry_run else MqttPublisher(config.mqtt)
    if publisher is not None:
        publisher.connect()

    initial_payload = collector.collect()
    schema_errors = validate_payload(initial_payload)
    if schema_errors:
        logger.warning("Schema validation failed with %s errors.", len(schema_errors))
        logger.debug("Schema errors: %s", schema_errors)
    else:
        logger.info("Schema validation passed.")
    initial_json = json.dumps(initial_payload, indent=2) if pretty_print else json.dumps(initial_payload)
    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as handle:
            handle.write(initial_json)
    if args.dry_run:
        logger.info("Dry run enabled; skipping MQTT publish.")
        logger.debug("Initial payload: %s", initial_json)
    elif publisher is not None:
        publisher.publish_discovery(initial_payload)
        publisher.publish(initial_json)

    if args.once:
        logger.info("Single-run mode enabled; exiting after initial payload.")
        if publisher is not None:
            publisher.disconnect()
        return

    interval = max(1, config.publish.interval_s)
    logging.info("Telemetry Tap started. Publishing every %s seconds.", interval)

    try:
        while True:
            payload = collector.collect()
            schema_errors = validate_payload(payload)
            if schema_errors:
                logger.warning(
                    "Schema validation failed with %s errors.", len(schema_errors)
                )
                logger.debug("Schema errors: %s", schema_errors)
            else:
                logger.debug("Schema validation passed.")
            payload_json = json.dumps(payload, indent=2) if pretty_print else json.dumps(payload)
            if args.dump_json:
                with open(args.dump_json, "w", encoding="utf-8") as handle:
                    handle.write(payload_json)
            if args.dry_run:
                logger.debug("Payload: %s", payload_json)
            elif publisher is not None:
                publisher.publish(payload_json)
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Telemetry Tap stopped.")
    finally:
        if publisher is not None:
            publisher.disconnect()


if __name__ == "__main__":
    main()
