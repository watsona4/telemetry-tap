# Changelog

## [1.2.1] - 2025-01-24

### Added
- Drive `total_rate_bps` metric - combined read + write transfer rate
- Computed directly in collector for simpler Home Assistant templates

## [1.2.0] - 2025-01-24

### Added
- Time server monitoring support (`enable_time_server` option)
- Chrony NTP tracking, sources, and server statistics
- GPS data collection via gpspipe
- PPS signal monitoring
- `pps_device` configuration option
- `gpsd-clients` and `chrony` packages in Docker image

### Changed
- Added `host_network: true` for access to chrony/gpsd services
- Added `/dev/pps0` device mapping

## [1.1.7] - 2025-01-23

### Fixed
- MQTT discovery sensor name duplication
- Schema validation for temp_c, security_issues, wear_leveling_pct
