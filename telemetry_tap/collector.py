from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import logging
import socket
import subprocess
from typing import Any
from urllib.request import urlopen

import psutil

from telemetry_tap.config import CollectorConfig
from telemetry_tap.logging_utils import TRACE_LEVEL

SCHEMA_NAME = "hwmon-exporter"
SCHEMA_VERSION = 1


@dataclass
class IoSnapshot:
    timestamp: float
    values: dict[str, psutil._common.sdiskio]


@dataclass
class NetSnapshot:
    timestamp: float
    values: dict[str, psutil._common.snetio]


@dataclass
class MetricsState:
    last_disk: IoSnapshot | None = None
    last_net: NetSnapshot | None = None
    drive_cache: dict[str, dict[str, Any]] = field(default_factory=dict)


class MetricsCollector:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self.state = MetricsState()
        self.logger = logging.getLogger(self.__class__.__name__)

    def collect(self) -> dict[str, Any]:
        self.logger.debug("Collecting metrics payload.")
        ts = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "ts": ts,
            "host": self._collect_host(ts),
            "health": self._collect_health(),
            "cpus": self._collect_cpus(),
            "memory": self._collect_memory(),
        }

        if filesystems := self._collect_filesystems():
            payload["filesystems"] = filesystems

        if drives := self._collect_drives():
            payload["drives"] = drives

        if ifaces := self._collect_ifaces():
            payload["ifaces"] = ifaces

        if batteries := self._collect_batteries():
            payload["batteries"] = batteries

        motherboard = self._collect_motherboard()
        if motherboard:
            payload["motherboard"] = motherboard

        gpus = self._collect_lhm_gpus()
        if gpus:
            payload["gpus"] = gpus

        self.logger.debug("Completed metrics payload collection.")
        return payload

    def _collect_host(self, ts: str) -> dict[str, Any]:
        boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        uptime_s = int(datetime.now(timezone.utc).timestamp() - boot.timestamp())
        return {
            "name": socket.gethostname(),
            "system": platform.system(),
            "os": platform.platform(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "boot_time": boot.isoformat(),
            "uptime_s": uptime_s,
        }

    def _collect_health(self) -> dict[str, Any]:
        issues: list[str] = []
        drive_health = self._drive_health_issues()
        issues.extend(drive_health)
        if issues:
            self.logger.warning("Health issues detected: %s", issues)
        return {"overall_ok": not issues, "issues": issues}

    def _collect_cpus(self) -> list[dict[str, Any]]:
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        logical_count = len(per_cpu)
        physical_count = psutil.cpu_count(logical=False) or logical_count
        num_cores = physical_count
        cores: list[dict[str, Any]] = []
        if physical_count and physical_count < logical_count:
            base = logical_count // physical_count
            remainder = logical_count % physical_count
            start = 0
            for core_idx in range(physical_count):
                size = base + (1 if core_idx < remainder else 0)
                threads = []
                for logical_idx in range(start, start + size):
                    threads.append(
                        {
                            "name": f"Thread {logical_idx}",
                            "load_pct": float(per_cpu[logical_idx]),
                        }
                    )
                cores.append({"name": f"Core {core_idx}", "threads": threads})
                start += size
        else:
            for idx, load in enumerate(per_cpu):
                cores.append(
                    {
                        "name": f"Core {idx}",
                        "threads": [
                            {"name": f"Thread {idx}", "load_pct": float(load)}
                        ],
                    }
                )

        cpu_entry: dict[str, Any] = {
            "name": "CPU",
            "num_cores": num_cores,
            "load_pct": float(psutil.cpu_percent(interval=None)),
            "cores": cores,
        }

        if hasattr(os, "getloadavg"):
            load_1m, load_5m, load_15m = os.getloadavg()
            cpu_entry.update(
                {
                    "load_1m": float(load_1m),
                    "load_5m": float(load_5m),
                    "load_15m": float(load_15m),
                }
            )

        temps = None
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures(fahrenheit=False)
        if temps:
            for entries in temps.values():
                if entries:
                    cpu_entry["temp_c"] = float(entries[0].current)
                    break
        else:
            self.logger.debug("No CPU temperature sensors found.")

        return [cpu_entry]

    def _collect_memory(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        memory: dict[str, Any] = {
            "system": {
                "used_b": int(vm.used),
                "available_b": int(vm.available),
                "load_pct": float(vm.percent),
            }
        }
        swap = psutil.swap_memory()
        if swap:
            memory["virtual"] = {
                "used_b": int(swap.used),
                "available_b": int(swap.free),
                "load_pct": float(swap.percent),
            }
        return memory

    def _collect_filesystems(self) -> list[dict[str, Any]]:
        filesystems: list[dict[str, Any]] = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                self.logger.debug("Skipping filesystem at %s (unreadable).", part.mountpoint)
                continue
            filesystems.append(
                {
                    "name": part.device,
                    "mountpoint": part.mountpoint,
                    "format": part.fstype,
                    "used_b": int(usage.used),
                    "available_b": int(usage.free),
                }
            )
        return filesystems

    def _collect_drives(self) -> list[dict[str, Any]]:
        metadata = self._drive_metadata()
        if not metadata:
            self.logger.debug("No drive metadata available.")
            return []

        io_stats = psutil.disk_io_counters(perdisk=True)
        now = datetime.now(timezone.utc).timestamp()
        drives: list[dict[str, Any]] = []
        for name, meta in metadata.items():
            if name not in io_stats:
                self.logger.debug("Drive %s missing IO stats.", name)
                continue
            io_entry = io_stats[name]
            entry: dict[str, Any] = {
                "name": name,
                "type": meta.get("type", "unknown"),
                "manufacturer": meta.get("manufacturer", "unknown"),
                "model": meta.get("model", "unknown"),
                "used_b": meta.get("used_b", 0),
                "available_b": meta.get("available_b", 0),
                "listed_cap_b": meta.get("listed_cap_b"),
                "serial_number": meta.get("serial"),
                "firmware_version": meta.get("firmware"),
                "total_read_b": int(io_entry.read_bytes),
                "total_write_b": int(io_entry.write_bytes),
            }

            if self.state.last_disk and name in self.state.last_disk.values:
                elapsed = now - self.state.last_disk.timestamp
                if elapsed > 0:
                    prev = self.state.last_disk.values[name]
                    entry["read_rate_bps"] = int(
                        (io_entry.read_bytes - prev.read_bytes) / elapsed
                    )
                    entry["write_rate_bps"] = int(
                        (io_entry.write_bytes - prev.write_bytes) / elapsed
                    )

            smart = meta.get("smart")
            if smart:
                entry["smart"] = smart

            drives.append(entry)

        self.state.last_disk = IoSnapshot(timestamp=now, values=io_stats)
        return drives

    def _collect_ifaces(self) -> list[dict[str, Any]]:
        addrs = psutil.net_if_addrs()
        io_stats = psutil.net_io_counters(pernic=True)
        now = datetime.now(timezone.utc).timestamp()
        ifaces: list[dict[str, Any]] = []

        for iface, addr_list in addrs.items():
            mac = None
            ipv4 = None
            ipv6 = None
            for addr in addr_list:
                if addr.family == socket.AF_INET:
                    ipv4 = addr.address
                elif addr.family == socket.AF_INET6:
                    ipv6 = addr.address.split("%", 1)[0]
                elif getattr(socket, "AF_LINK", None) == addr.family:
                    mac = addr.address
                elif getattr(psutil, "AF_LINK", None) == addr.family:
                    mac = addr.address
            if mac is None:
                self.logger.debug("Skipping interface %s without MAC.", iface)
                continue

            entry: dict[str, Any] = {
                "name": iface,
                "mac": mac,
                "ipv4": ipv4,
                "ipv6": ipv6,
            }

            if iface in io_stats:
                counters = io_stats[iface]
                entry["data_up_b"] = int(counters.bytes_sent)
                entry["data_down_b"] = int(counters.bytes_recv)

                if self.state.last_net and iface in self.state.last_net.values:
                    elapsed = now - self.state.last_net.timestamp
                    if elapsed > 0:
                        prev = self.state.last_net.values[iface]
                        entry["rate_up_bps"] = int(
                            (counters.bytes_sent - prev.bytes_sent) / elapsed
                        )
                        entry["rate_down_bps"] = int(
                            (counters.bytes_recv - prev.bytes_recv) / elapsed
                        )

            ifaces.append(entry)

        self.state.last_net = NetSnapshot(timestamp=now, values=io_stats)
        return ifaces

    def _collect_batteries(self) -> list[dict[str, Any]]:
        if not hasattr(psutil, "sensors_battery"):
            self.logger.debug("Battery sensors not supported on this platform.")
            return []
        battery = psutil.sensors_battery()
        if battery is None:
            self.logger.debug("No battery data available.")
            return []
        return [
            {
                "name": "Battery",
                "discharging": battery.power_plugged is False,
                "charge_level_pct": float(battery.percent),
            }
        ]

    def _drive_health_issues(self) -> list[str]:
        issues: list[str] = []
        metadata = self.state.drive_cache
        for name, meta in metadata.items():
            smart = meta.get("smart")
            if not smart:
                continue
            health = smart.get("overall_health")
            if health and health.upper() not in {"PASSED", "OK"}:
                issues.append(f"Drive {name} SMART health: {health}")
        return issues

    def _drive_metadata(self) -> dict[str, dict[str, Any]]:
        if self.state.drive_cache:
            return self.state.drive_cache

        metadata: dict[str, dict[str, Any]] = {}
        metadata.update(self._drive_metadata_lsblk())
        self._populate_usage(metadata)
        self._populate_smart(metadata)
        self.state.drive_cache = metadata
        return metadata

    def _drive_metadata_lsblk(self) -> dict[str, dict[str, Any]]:
        if platform.system().lower() != "linux":
            self.logger.debug("Skipping lsblk drive metadata on non-Linux.")
            return {}
        output = self._run_command([self.config.lsblk_path, "-J", "-O"])
        if output is None:
            self.logger.debug("lsblk command failed or missing.")
            return {}
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse lsblk output JSON.")
            return {}

        metadata: dict[str, dict[str, Any]] = {}
        for block in data.get("blockdevices", []):
            if block.get("type") != "disk":
                continue
            name = block.get("name")
            if not name:
                continue
            drive_type = "HDD" if block.get("rota") else "SSD"
            metadata[name] = {
                "manufacturer": block.get("vendor") or "unknown",
                "model": block.get("model") or "unknown",
                "type": drive_type,
                "listed_cap_b": block.get("size")
                if isinstance(block.get("size"), int)
                else None,
            }
        return metadata

    def _populate_usage(self, metadata: dict[str, dict[str, Any]]) -> None:
        if not metadata:
            return
        partitions = psutil.disk_partitions(all=False)
        for part in partitions:
            device = Path(part.device).name
            if device not in metadata:
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                self.logger.debug("Skipping usage for %s (unreadable).", part.mountpoint)
                continue
            metadata[device]["used_b"] = int(usage.used)
            metadata[device]["available_b"] = int(usage.free)

    def _populate_smart(self, metadata: dict[str, dict[str, Any]]) -> None:
        if not metadata:
            return
        for name in metadata:
            device_path = f"/dev/{name}"
            smart = self._smartctl_info(device_path)
            if smart:
                metadata[name]["smart"] = smart
            else:
                self.logger.debug("SMART data unavailable for %s.", device_path)

    def _smartctl_info(self, device_path: str) -> dict[str, Any] | None:
        output = self._run_command(
            [self.config.smartctl_path, "-a", "-j", device_path], stderr=subprocess.DEVNULL
        )
        if output is None:
            self.logger.debug("smartctl failed for %s.", device_path)
            return None
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse smartctl JSON for %s.", device_path)
            return None
        smart: dict[str, Any] = {}
        if "smart_status" in data:
            smart["overall_health"] = (
                "PASSED" if data["smart_status"].get("passed") else "FAILED"
            )
        nvme = data.get("nvme_smart_health_information_log")
        if nvme:
            mapping = {
                "power_on_hours": nvme.get("power_on_hours"),
                "critical_warning": nvme.get("critical_warning"),
                "available_spare_pct": nvme.get("available_spare"),
                "available_spare_threshold_pct": nvme.get(
                    "available_spare_threshold"
                ),
                "percentage_used_pct": nvme.get("percentage_used"),
                "data_units_read": nvme.get("data_units_read"),
                "data_units_written": nvme.get("data_units_written"),
                "host_read_commands": nvme.get("host_reads"),
                "host_write_commands": nvme.get("host_writes"),
                "controller_busy_time": nvme.get("controller_busy_time"),
                "power_cycles": nvme.get("power_cycles"),
                "unsafe_shutdowns": nvme.get("unsafe_shutdowns"),
                "media_errors": nvme.get("media_errors"),
                "num_err_log_entries": nvme.get("num_err_log_entries"),
            }
            for key, value in mapping.items():
                if value is not None:
                    smart[key] = value
        if smart:
            return smart
        return None

    def _collect_motherboard(self) -> dict[str, Any] | None:
        motherboard: dict[str, Any] = {}

        lhm_board = self._collect_lhm_board()
        if lhm_board:
            self._merge_motherboard(motherboard, lhm_board)

        sensors_board = self._collect_linux_sensors()
        if sensors_board:
            self._merge_motherboard(motherboard, sensors_board)

        dmi_board = self._collect_dmidecode()
        if dmi_board:
            motherboard.update(dmi_board)

        return motherboard if motherboard else None

    def _merge_motherboard(
        self, target: dict[str, Any], incoming: dict[str, Any]
    ) -> None:
        for key, value in incoming.items():
            if isinstance(value, list):
                target.setdefault(key, [])
                target[key].extend(value)
            else:
                target.setdefault(key, value)

    def _collect_lhm_board(self) -> dict[str, Any] | None:
        data = self._read_lhm()
        if not data:
            return None
        motherboard: dict[str, Any] = {}
        temps: list[dict[str, Any]] = []
        fans: list[dict[str, Any]] = []
        voltages: list[dict[str, Any]] = []
        powers: list[dict[str, Any]] = []

        for sensor in data.get("motherboard_sensors", []):
            category = sensor.get("category")
            name = sensor.get("name")
            value = sensor.get("value")
            if name is None or value is None:
                continue
            if category == "temperature":
                temps.append({"name": name, "temp_c": value, "source": "lhm"})
            elif category == "fan":
                fans.append({"name": name, "rpm": value, "source": "lhm"})
            elif category == "voltage":
                voltages.append(
                    {"name": name, "voltage_v": value, "source": "lhm"}
                )
            elif category == "power":
                powers.append({"name": name, "power_w": value, "source": "lhm"})

        if temps:
            motherboard["temps"] = temps
        if fans:
            motherboard["fans"] = fans
        if voltages:
            motherboard["voltages"] = voltages
        if powers:
            motherboard["powers"] = powers

        return motherboard if motherboard else None

    def _collect_linux_sensors(self) -> dict[str, Any] | None:
        if platform.system().lower() != "linux":
            return None
        output = self._run_command([self.config.sensors_path, "-j"])
        if output is None:
            return None
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse sensors JSON output.")
            return None

        temps: list[dict[str, Any]] = []
        fans: list[dict[str, Any]] = []
        voltages: list[dict[str, Any]] = []
        currents: list[dict[str, Any]] = []
        powers: list[dict[str, Any]] = []

        for chip, chip_data in data.items():
            if not isinstance(chip_data, dict):
                continue
            for label, readings in chip_data.items():
                if not isinstance(readings, dict):
                    continue
                for key, value in readings.items():
                    if not isinstance(value, (int, float)):
                        continue
                    entry_name = f"{chip} {label}"
                    if key.endswith("_input"):
                        if key.startswith("temp"):
                            temps.append(
                                {
                                    "name": entry_name,
                                    "temp_c": float(value),
                                    "source": "sensors",
                                }
                            )
                        elif key.startswith("fan"):
                            fans.append(
                                {
                                    "name": entry_name,
                                    "rpm": float(value),
                                    "source": "sensors",
                                }
                            )
                        elif key.startswith("in"):
                            voltages.append(
                                {
                                    "name": entry_name,
                                    "voltage_v": float(value),
                                    "source": "sensors",
                                }
                            )
                        elif key.startswith("curr"):
                            currents.append(
                                {
                                    "name": entry_name,
                                    "current_a": float(value),
                                    "source": "sensors",
                                }
                            )
                        elif key.startswith("power"):
                            powers.append(
                                {
                                    "name": entry_name,
                                    "power_w": float(value),
                                    "source": "sensors",
                                }
                            )

        motherboard: dict[str, Any] = {}
        if temps:
            motherboard["temps"] = temps
        if fans:
            motherboard["fans"] = fans
        if voltages:
            motherboard["voltages"] = voltages
        if currents:
            motherboard["currents"] = currents
        if powers:
            motherboard["powers"] = powers

        return motherboard if motherboard else None

    def _collect_dmidecode(self) -> dict[str, Any] | None:
        if platform.system().lower() != "linux":
            return None
        output = self._run_command([self.config.dmidecode_path, "-t", "baseboard", "-t", "bios"])
        if output is None:
            return None
        manufacturer = None
        name = None
        bios_version = None
        bios_date = None
        in_baseboard = False
        in_bios = False
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("Base Board Information"):
                in_baseboard = True
                in_bios = False
                continue
            if stripped.startswith("BIOS Information"):
                in_bios = True
                in_baseboard = False
                continue
            if ":" not in stripped:
                continue
            key, value = [part.strip() for part in stripped.split(":", 1)]
            if in_baseboard:
                if key == "Manufacturer":
                    manufacturer = value
                elif key in {"Product Name", "Product"}:
                    name = value
            if in_bios:
                if key == "Version":
                    bios_version = value
                elif key == "Release Date":
                    bios_date = value

        motherboard = {}
        if name:
            motherboard["name"] = name
        if manufacturer:
            motherboard["manufacturer"] = manufacturer
        if bios_version:
            motherboard["bios_version"] = bios_version
        if bios_date:
            motherboard["bios_date"] = bios_date
        return motherboard if motherboard else None

    def _collect_lhm_gpus(self) -> list[dict[str, Any]]:
        data = self._read_lhm()
        if not data:
            return []
        gpus: list[dict[str, Any]] = []
        for device in data.get("gpus", []):
            name = device.get("name")
            if not name:
                continue
            core_load = device.get("core_load")
            engine_loads = device.get("engines", [])
            core = {
                "name": "Core",
                "load_pct": core_load if core_load is not None else 0,
            }
            engines = [
                {"name": engine["name"], "load_pct": engine["load"]}
                for engine in engine_loads
                if "name" in engine and "load" in engine
            ]
            if not engines:
                engines = [{"name": "Core", "load_pct": core["load_pct"]}]
            entry: dict[str, Any] = {"name": name, "core": core, "engines": engines}
            if device.get("temp_c") is not None:
                entry["temp_c"] = device["temp_c"]
            gpus.append(entry)
        return gpus

    def _read_lhm(self) -> dict[str, Any] | None:
        if not self.config.librehardwaremonitor_url:
            self.logger.debug("LibreHardwareMonitor URL not configured.")
            return None
        try:
            with urlopen(self.config.librehardwaremonitor_url, timeout=2) as response:
                payload = response.read().decode("utf-8")
        except OSError:
            self.logger.debug("Failed to fetch LibreHardwareMonitor JSON.")
            return None
        if self.logger.isEnabledFor(TRACE_LEVEL):
            self.logger.log(TRACE_LEVEL, "LibreHardwareMonitor raw payload: %s", payload)
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse LibreHardwareMonitor JSON.")
            return None
        return self._parse_lhm(raw)

    def _run_command(
        self, command: list[str], stderr: int | None = None
    ) -> str | None:
        try:
            if stderr is None:
                result = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    capture_output=True,
                )
            else:
                result = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                )
        except FileNotFoundError:
            self.logger.debug("Command not found: %s", command[0])
            return None
        if result.returncode != 0:
            self.logger.debug(
                "Command failed (%s): %s", result.returncode, " ".join(command)
            )
            if result.stderr:
                self.logger.log(TRACE_LEVEL, "stderr: %s", result.stderr.strip())
            return None
        if result.stdout:
            self.logger.log(TRACE_LEVEL, "stdout: %s", result.stdout.strip())
        return result.stdout

    def _parse_lhm(self, raw: dict[str, Any]) -> dict[str, Any]:
        motherboard_sensors: list[dict[str, Any]] = []
        gpu_sensors: dict[str, list[dict[str, Any]]] = {}
        gpus: list[dict[str, Any]] = []

        def walk(
            node: dict[str, Any],
            hardware_type: str | None,
            gpu_name: str | None,
        ) -> None:
            node_type = node.get("Type")
            node_text = node.get("Text") or ""
            next_hardware = hardware_type
            next_gpu_name = gpu_name
            if node_type == "Hardware":
                raw_hardware = (node.get("HardwareType") or "").lower()
                if raw_hardware.startswith("gpu"):
                    next_hardware = "gpu"
                elif raw_hardware in {"motherboard", "mainboard"}:
                    next_hardware = "motherboard"
                else:
                    next_hardware = raw_hardware or hardware_type
                if next_hardware == "gpu":
                    gpus.append({"name": node_text})
                    gpu_sensors.setdefault(node_text, [])
                    next_gpu_name = node_text
                for child in node.get("Children", []):
                    walk(child, next_hardware, next_gpu_name)
                return
            if node_type == "Sensor":
                sensor_type = (node.get("SensorType") or "").lower()
                value = node.get("Value")
                if value is None:
                    return
                sensor_entry = {
                    "category": sensor_type,
                    "name": node_text,
                    "value": float(value),
                }
                if hardware_type == "motherboard":
                    motherboard_sensors.append(sensor_entry)
                if hardware_type == "gpu":
                    if gpu_name:
                        gpu_sensors.setdefault(gpu_name, []).append(sensor_entry)
                return
            for child in node.get("Children", []):
                walk(child, next_hardware, next_gpu_name)

        for root in raw.get("Children", []):
            walk(root, None, None)

        parsed_gpus: list[dict[str, Any]] = []
        for gpu in gpus:
            name = gpu.get("name")
            sensors = gpu_sensors.get(name, [])
            engines = [
                {"name": sensor["name"], "load": sensor["value"]}
                for sensor in sensors
                if sensor["category"] == "load"
            ]
            temp = next(
                (
                    sensor["value"]
                    for sensor in sensors
                    if sensor["category"] == "temperature"
                ),
                None,
            )
            parsed_gpus.append(
                {
                    "name": name,
                    "engines": engines,
                    "core_load": next(
                        (
                            engine["load"]
                            for engine in engines
                            if "core" in engine["name"].lower()
                        ),
                        None,
                    ),
                    "temp_c": temp,
                }
            )

        return {"motherboard_sensors": motherboard_sensors, "gpus": parsed_gpus}
