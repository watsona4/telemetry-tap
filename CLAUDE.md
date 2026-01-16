# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telemetry Tap is a cross-platform hardware monitoring tool that collects system metrics from various sources, normalizes them into a consistent JSON schema, and publishes them via MQTT. It supports integration with Home Assistant through MQTT discovery.

## Commands

### Installation and Setup
```bash
# Install the package in editable mode (requires Python >= 3.10)
pip install -e .

# Copy and edit configuration
cp config/example.cfg config/local.cfg
```

### Running the Application
```bash
# Standard run with config file
telemetry-tap --config config/local.cfg --log-level INFO

# Dry run mode (no MQTT publishing, useful for testing)
telemetry-tap --config config/local.cfg --dry-run -v

# Single run mode (collect once and exit)
telemetry-tap --config config/local.cfg --once

# Dump JSON payload to file for inspection
telemetry-tap --config config/local.cfg --dump-json telemetry.json

# Debug mode (pretty-print payloads, colorized logs)
telemetry-tap --config config/local.cfg -v

# Trace mode (includes CLI tool output)
telemetry-tap --config config/local.cfg -vv
```

### Development Commands
```bash
# Run with verbose logging for debugging
telemetry-tap --config config/local.cfg -v --dry-run

# Validate schema output
telemetry-tap --config config/local.cfg --once --dump-json out.json
```

### Testing
```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=telemetry_tap --cov-report=term-missing

# Run only Windows-specific tests
pytest -m windows

# Run only Linux-specific tests
pytest -m linux

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/test_collector_windows.py

# Run with verbose output
pytest -v

# Run and show print statements
pytest -s
```

## Architecture

### Core Components

**Main Entry Point (`main.py`)**
- Parses CLI arguments and configures logging
- Orchestrates the collection and publishing loop
- Handles `--dry-run`, `--once`, and `--dump-json` modes
- Validates payloads against JSON schema before publishing

**Metrics Collector (`collector.py`)**
- Central class: `MetricsCollector` - responsible for gathering all hardware metrics
- Maintains state between collections via `MetricsState` (for calculating rates like disk I/O, network throughput)
- Collects metrics from multiple sources:
  - **psutil**: Cross-platform CPU, memory, disk, network, battery data
  - **LibreHardwareMonitor (LHM)**: Windows-specific hardware monitoring via JSON endpoint
  - **Linux tools**: `lsblk`, `smartctl`, `sensors`, `dmidecode` for enhanced metadata
- Implements health monitoring for systemd services and Docker containers

**MQTT Publisher (`mqtt_client.py`)**
- `MqttPublisher` class handles MQTT connectivity and publishing
- Publishes two types of messages:
  1. Regular telemetry payloads to `base_topic`
  2. Home Assistant discovery payloads to `discovery_topic`
- Supports TLS, authentication, QoS, and retain settings

**Configuration (`config.py`)**
- Uses dataclasses for type-safe configuration
- Loads from CFG/INI format via `configparser`
- Four main config sections: `MqttConfig`, `PublishConfig`, `CollectorConfig`, `HealthConfig`

**Schema Validation (`schema.py`)**
- Validates payloads against `telemetry_tap/schemas/hwmon-exporter.schema.json`
- Uses JSON Schema Draft 2020-12 via `jsonschema` library
- Returns list of validation errors for debugging

### Data Collection Flow

1. **State Management**: `MetricsCollector` maintains `MetricsState` with:
   - Last disk I/O snapshot for calculating read/write rates
   - Last network snapshot for calculating bandwidth rates
   - Drive metadata cache to avoid re-querying static information

2. **Multi-Source Collection**: Metrics are collected from:
   - psutil (always available)
   - LibreHardwareMonitor JSON endpoint (Windows, optional)
   - Linux CLI tools (optional, graceful degradation if missing)

3. **Data Merging**: Multiple sources can contribute to the same metric:
   - Battery: psutil provides charge level/discharging state, LHM provides voltage/current/power
   - CPU: psutil provides load, LHM provides temperature/voltage/per-core metrics
   - Drives: lsblk provides metadata, psutil provides I/O stats, smartctl provides SMART data, LHM provides temperature

4. **Schema Output**: Final payload follows `hwmon-exporter` schema v1 with structure:
   ```
   {
     "schema": {"name": "hwmon-exporter", "version": 1},
     "ts": "ISO8601 timestamp",
     "host": {...},
     "health": {...},
     "cpus": [...],
     "memory": {...},
     "filesystems": [...],  // optional
     "drives": [...],       // optional
     "ifaces": [...],       // optional
     "batteries": [...],    // optional
     "motherboard": {...},  // optional
     "gpus": [...]         // optional
   }
   ```

### LibreHardwareMonitor Integration

LHM data parsing has two modes detected automatically:
- **Legacy format**: Walks tree structure looking for Hardware/Sensor nodes
- **New format** (with `SensorId`): Uses `_parse_lhm_tree()` for better structured parsing

The `_parse_lhm()` and `_parse_lhm_tree()` methods handle both formats, extracting:
- CPU core-level metrics (voltage, power, clock, temperature)
- GPU metrics per engine
- Battery data (charge, voltage, current, power)
- Drive temperature and activity
- Motherboard sensors (temps, fans, voltages, currents, power)

### Health Monitoring

The health system aggregates issues from:
- **Threshold checks**: CPU >90%, memory >90%, swap >90%, disk space >90%
- **Drive SMART**: Overall health status from `smartctl`
- **Services**: systemd service status on Linux
- **Containers**: Docker container status (running/healthy/exited)

All issues are consolidated into `health.issues` array with a human-readable summary.

## Key Design Patterns

**Graceful Degradation**: Missing tools or data sources don't cause failures. The collector checks for availability and logs debug messages when tools are missing.

**Caching**: Drive metadata (manufacturer, model, SMART data) is cached in `MetricsState.drive_cache` since it rarely changes.

**Rate Calculation**: Disk I/O and network rates are calculated by comparing current counters with previous snapshots stored in state.

**Platform Detection**: Uses `platform.system().lower()` to determine OS and conditionally enable Linux-specific collectors.

**Partition Mapping**: Filesystems are enriched with their backing block device by building a partition map from drive metadata.

## Configuration Notes

- MQTT connection is required unless using `--dry-run`
- LibreHardwareMonitor URL is optional (Windows-only enhancement)
- Linux tools (`lsblk`, `smartctl`, `sensors`, `dmidecode`) are optional
- Health monitoring sections (`services`, `containers`) are optional
- Publish interval minimum is 1 second (enforced in code)

## Common Patterns

When adding new collectors or metrics:
1. Add data source query in `collector.py` (follow existing `_collect_*` pattern)
2. Update `hwmon-exporter.schema.json` if adding new fields
3. Merge data from multiple sources when available (see battery/CPU examples)
4. Use debug logging for missing tools/data, not warnings/errors
5. Return empty lists/dicts when data is unavailable (never fail)
