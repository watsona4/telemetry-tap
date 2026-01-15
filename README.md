# telemetry-tap
Telemetry Tap is a lightweight, universal hardware monitoring script that collects sensor + system stats, normalizes them, and publishes in a consistent format for dashboards and alerts. Designed for low overhead and portability; integrates cleanly with MQTT/Prometheus/Home Assistant.

## Features
- Cross-platform metrics collection with `psutil` (CPU, memory, disks, network, batteries).
- Optional Linux enrichment via `lsblk` and `smartctl` when available.
- Optional LibreHardwareMonitor support via its JSON endpoint (useful on Windows).
- Publishes JSON payloads matching the supplied hardware schema over MQTT.
- Publishes a Home Assistant MQTT discovery payload on startup.
- Validates payloads against the bundled JSON schema before publishing.

## Quick start
1. Install dependencies:
   ```bash
   pip install -e .
   ```
2. Copy and edit the example config:
   ```bash
   cp config/example.cfg config/local.cfg
   ```
3. Run the exporter:
   ```bash
   telemetry-tap --config config/local.cfg --log-level INFO
   ```
   For diagnostics without publishing over MQTT, use `--dry-run` and `-v`. Use `-vv` to enable trace logging of CLI tool output.

## Configuration
The exporter reads a CFG/INI style configuration file. See `config/example.cfg` for defaults.

### MQTT
```ini
[mqtt]
host = localhost
port = 1883
base_topic = telemetry/hwmon
discovery_topic = homeassistant
client_id = telemetry-tap
username =
password =
qos = 0
retain = false
```

### Publish interval
```ini
[publish]
interval_s = 15
```

### Optional collectors
```ini
[collector]
smartctl_path = smartctl
lsblk_path = lsblk
sensors_path = sensors
dmidecode_path = dmidecode
librehardwaremonitor_url = http://localhost:8085/data.json
```

LibreHardwareMonitor must expose its JSON endpoint (default is `http://localhost:8085/data.json`). On Linux, `lsblk`, `smartctl`, `sensors`, and `dmidecode` are used when present to enrich drive metadata, SMART stats, and motherboard details.

Home Assistant discovery is published to the configured `discovery_topic` with a single sensor that exposes the full payload as attributes and reports host uptime as its state.
