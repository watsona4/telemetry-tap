#!/usr/bin/with-contenv bashio

# Read configuration from add-on options
MQTT_HOST=$(bashio::config 'mqtt_host')
MQTT_PORT=$(bashio::config 'mqtt_port')
MQTT_USERNAME=$(bashio::config 'mqtt_username')
MQTT_PASSWORD=$(bashio::config 'mqtt_password')
MQTT_BASE_TOPIC=$(bashio::config 'mqtt_base_topic')
MQTT_DISCOVERY_TOPIC=$(bashio::config 'mqtt_discovery_topic')
MQTT_CLIENT_ID=$(bashio::config 'mqtt_client_id')
PUBLISH_INTERVAL=$(bashio::config 'publish_interval')
LOG_LEVEL=$(bashio::config 'log_level')

# Read containers list and convert to comma-separated string
CONTAINERS=""
for container in $(bashio::config 'containers'); do
    if [ -n "${CONTAINERS}" ]; then
        CONTAINERS="${CONTAINERS},${container}"
    else
        CONTAINERS="${container}"
    fi
done

# Generate config file
CONFIG_FILE="/tmp/telemetry-tap.cfg"

cat > "${CONFIG_FILE}" <<EOF
[mqtt]
host = ${MQTT_HOST}
port = ${MQTT_PORT}
username = ${MQTT_USERNAME}
password = ${MQTT_PASSWORD}
base_topic = ${MQTT_BASE_TOPIC}
discovery_topic = ${MQTT_DISCOVERY_TOPIC}
client_id = ${MQTT_CLIENT_ID}
qos = 1
retain = true

[publish]
interval_s = ${PUBLISH_INTERVAL}

[collector]
# Tool paths for Alpine Linux
smartctl_path = smartctl
lsblk_path = lsblk
sensors_path = sensors
dmidecode_path = dmidecode

[health]
enable_hassio = true
containers = ${CONTAINERS}
EOF

bashio::log.info "Starting Telemetry Tap..."
bashio::log.info "MQTT broker: ${MQTT_HOST}:${MQTT_PORT}"
bashio::log.info "Publishing to: ${MQTT_BASE_TOPIC}"

# Run telemetry-tap
exec telemetry-tap --config "${CONFIG_FILE}" --log-level "${LOG_LEVEL}"
