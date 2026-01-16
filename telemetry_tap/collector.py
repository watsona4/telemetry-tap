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

from telemetry_tap.config import CollectorConfig, HealthConfig
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
    def __init__(self, config: CollectorConfig, health: HealthConfig) -> None:
        self.config = config
        self.health = health
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
        issues.extend(self._drive_health_issues())
        issues.extend(self._threshold_issues())
        services = self._collect_services()
        containers = self._collect_containers()
        if services:
            for service in services:
                if not service["ok"]:
                    issues.append(f"Service {service['name']} status {service['status']}")
        if containers:
            for container in containers:
                if not container["ok"]:
                    issues.append(
                        f"Container {container['name']} status {container['status']}"
                    )
        summary = "ok" if not issues else f"issues: {'; '.join(issues)}"
        issues_with_summary = [summary] + issues if issues else [summary]
        if issues:
            self.logger.warning("Health issues detected: %s", issues_with_summary)
        health: dict[str, Any] = {
            "overall_ok": not issues,
            "issues": issues_with_summary,
        }
        if services:
            health["services"] = services
        if containers:
            health["containers"] = containers
        return health

    def _collect_cpus(self) -> list[dict[str, Any]]:
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        logical_count = len(per_cpu)
        physical_count = psutil.cpu_count(logical=False) or logical_count
        num_physical_cores = physical_count
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
            "num_physical_cores": num_physical_cores,
            "num_logical_cores": logical_count,
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

        lhm_cpu = self._collect_lhm_cpu()
        if lhm_cpu:
            for key in ["temp_c", "voltage_core_v", "voltage_soc_v"]:
                if key in lhm_cpu and lhm_cpu[key] is not None:
                    cpu_entry[key] = lhm_cpu[key]
            core_metrics = lhm_cpu.get("cores", {})
            for core in cores:
                index = int(core["name"].split()[-1])
                lhm_index = index + 1
                if lhm_index in core_metrics:
                    core.update(core_metrics[lhm_index])

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
        drive_metadata = self.state.drive_cache or self._drive_metadata()
        partition_map: dict[str, str] = {}
        for drive_name, meta in drive_metadata.items():
            for partition in meta.get("partitions", []):
                partition_name = partition.get("name")
                if partition_name:
                    partition_map[partition_name] = drive_name
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                self.logger.debug("Skipping filesystem at %s (unreadable).", part.mountpoint)
                continue
            entry = {
                "name": part.device,
                "mountpoint": part.mountpoint,
                "format": part.fstype,
                "used_b": int(usage.used),
                "available_b": int(usage.free),
            }
            drive_name = partition_map.get(part.device)
            if drive_name:
                entry["backing_blockdev"] = {
                    "device": part.device,
                    "drive_name": drive_name,
                    "partition_name": part.device,
                }
            filesystems.append(entry)
        return filesystems

    def _collect_drives(self) -> list[dict[str, Any]]:
        metadata = self._drive_metadata()
        lhm_data = self._read_lhm()
        lhm_drives = self._collect_lhm_drives(lhm_data)
        if not metadata:
            if lhm_drives:
                self.logger.debug("Using LibreHardwareMonitor drive data fallback.")
                return [
                    {
                        "name": name,
                        "type": "unknown",
                        "manufacturer": "unknown",
                        "model": name,
                        "used_b": 0,
                        "available_b": 0,
                        **values,
                    }
                    for name, values in lhm_drives.items()
                ]
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
            if name in lhm_drives:
                for key in [
                    "temp_c",
                    "read_activity_pct",
                    "write_activity_pct",
                    "read_rate_bps",
                    "write_rate_bps",
                    "total_read_b",
                    "total_write_b",
                ]:
                    if key in lhm_drives[name] and lhm_drives[name][key] is not None:
                        entry[key] = lhm_drives[name][key]

            drives.append(entry)

        self.state.last_disk = IoSnapshot(timestamp=now, values=io_stats)
        return drives

    def _collect_ifaces(self) -> list[dict[str, Any]]:
        addrs = psutil.net_if_addrs()
        io_stats = psutil.net_io_counters(pernic=True)
        iface_stats = psutil.net_if_stats()
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
                else:
                    entry["rate_up_bps"] = 0
                    entry["rate_down_bps"] = 0

                speed_mbps = iface_stats.get(iface).speed if iface in iface_stats else 0
                if speed_mbps:
                    link_bps = (speed_mbps * 1_000_000) / 8
                    utilization = (
                        max(entry["rate_up_bps"], entry["rate_down_bps"]) / link_bps
                    ) * 100
                    entry["utilization_pct"] = min(max(utilization, 0.0), 100.0)

            ifaces.append(entry)

        self.state.last_net = NetSnapshot(timestamp=now, values=io_stats)
        return ifaces

    def _threshold_issues(self) -> list[str]:
        issues: list[str] = []
        cpu_pct = psutil.cpu_percent(interval=None)
        if cpu_pct > 90:
            issues.append(f"High CPU utilization {cpu_pct:.1f}%")
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            issues.append(f"High memory utilization {mem.percent:.1f}%")
        swap = psutil.swap_memory()
        if swap and swap.percent > 90:
            issues.append(f"High swap utilization {swap.percent:.1f}%")
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            pct = (usage.used / usage.total) * 100 if usage.total else 0
            if pct > 90:
                issues.append(f"Low disk space on {part.mountpoint} ({pct:.1f}%)")
        return issues

    def _collect_services(self) -> list[dict[str, Any]]:
        if not self.health.services:
            return []
        if platform.system().lower() != "linux":
            self.logger.debug("Service checks only supported on Linux/systemd.")
            return []
        services: list[dict[str, Any]] = []
        for service in self.health.services:
            output = self._run_command(
                [
                    "systemctl",
                    "show",
                    service,
                    "--no-page",
                    "--property=ActiveState,SubState,LoadState",
                ]
            )
            if output is None:
                services.append(
                    {
                        "name": service,
                        "ok": False,
                        "status": "failed",
                        "detail": "systemctl unavailable or failed",
                        "loaded": "error",
                    }
                )
                continue
            fields = dict(
                line.split("=", 1)
                for line in output.splitlines()
                if "=" in line
            )
            active = fields.get("ActiveState", "unknown")
            sub = fields.get("SubState", "unknown")
            loaded = fields.get("LoadState", "unknown")
            status = active if active in {"active", "inactive", "failed"} else sub
            ok = active == "active"
            services.append(
                {
                    "name": service,
                    "ok": ok,
                    "status": status,
                    "loaded": loaded,
                    "detail": None if ok else f"{active}/{sub}",
                }
            )
        return services

    def _collect_containers(self) -> list[dict[str, Any]]:
        if not self.health.containers:
            return []
        output = self._run_command(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.Names}}||{{.Status}}||{{.Image}}",
            ]
        )
        if output is None:
            self.logger.debug("Docker not available for container checks.")
            return [
                {
                    "name": name,
                    "ok": False,
                    "status": "unknown",
                    "detail": "docker not available",
                }
                for name in self.health.containers
            ]
        status_map = {}
        for line in output.splitlines():
            name, status, image = (part.strip() for part in line.split("||"))
            status_map[name] = {"status": status, "image": image}
        containers: list[dict[str, Any]] = []
        for name in self.health.containers:
            info = status_map.get(name)
            if not info:
                containers.append(
                    {
                        "name": name,
                        "ok": False,
                        "status": "exited",
                        "detail": "not found",
                    }
                )
                continue
            status_raw = info["status"]
            status_lower = status_raw.lower()
            if status_lower.startswith("up"):
                status = "running"
                ok = "unhealthy" not in status_lower
                if "healthy" in status_lower:
                    status = "healthy"
            elif status_lower.startswith("exited"):
                status = "exited"
                ok = False
            else:
                status = "unknown"
                ok = False
            containers.append(
                {
                    "name": name,
                    "ok": ok,
                    "status": status,
                    "image": info.get("image"),
                    "detail": status_raw,
                }
            )
        return containers

    @staticmethod
    def _parse_core_index(label: str) -> int | None:
        for part in label.replace("#", " ").split():
            if part.isdigit():
                return int(part)
        return None

    def _collect_batteries(self) -> list[dict[str, Any]]:
        batteries: list[dict[str, Any]] = []
        psutil_battery: dict[str, Any] | None = None
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery is not None:
                psutil_battery = {
                    "name": "Battery",
                    "discharging": battery.power_plugged is False,
                    "charge_level_pct": float(battery.percent),
                }
            else:
                self.logger.debug("No battery data available from psutil.")
        else:
            self.logger.debug("Battery sensors not supported on this platform.")

        lhm_data = self._read_lhm()
        lhm_batteries = self._collect_lhm_batteries(lhm_data)
        if psutil_battery and lhm_batteries:
            lhm_primary = lhm_batteries[0].copy()
            lhm_primary["discharging"] = psutil_battery["discharging"]
            lhm_primary["charge_level_pct"] = psutil_battery["charge_level_pct"]
            batteries.append(lhm_primary)
        elif psutil_battery:
            batteries.append(psutil_battery)
        else:
            batteries.extend(lhm_batteries)

        return batteries

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

        # On Windows, create basic metadata from IO stats
        if not metadata and platform.system().lower() == "windows":
            metadata.update(self._drive_metadata_windows())

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
            partitions = self._parse_lsblk_partitions(block.get("children", []))
            metadata[name] = {
                "manufacturer": block.get("vendor") or "unknown",
                "model": block.get("model") or "unknown",
                "type": drive_type,
                "listed_cap_b": block.get("size")
                if isinstance(block.get("size"), int)
                else None,
                "partitions": partitions,
            }
        return metadata

    def _drive_metadata_windows(self) -> dict[str, dict[str, Any]]:
        """Create basic drive metadata on Windows from IO stats."""
        io_stats = psutil.disk_io_counters(perdisk=True)
        if not io_stats:
            return {}

        metadata: dict[str, dict[str, Any]] = {}
        for drive_name in io_stats.keys():
            metadata[drive_name] = {
                "manufacturer": "unknown",
                "model": drive_name,
                "type": "unknown",
            }

        self.logger.debug("Created Windows drive metadata for %d drives", len(metadata))
        return metadata

    def _parse_lsblk_partitions(self, children: list[dict[str, Any]]) -> list[dict[str, Any]]:
        partitions: list[dict[str, Any]] = []
        for child in children or []:
            if child.get("type") != "part":
                continue
            name = child.get("name")
            if not name:
                continue
            content = "filesystem"
            fstype = child.get("fstype")
            if child.get("type") == "crypt" or fstype in {"crypto_LUKS"}:
                content = "encrypted"
            if fstype in {"swap"}:
                content = "swap"
            partition: dict[str, Any] = {
                "name": child.get("path") or f"/dev/{name}",
                "number": child.get("partno"),
                "type_guid": child.get("parttype"),
                "label": child.get("partlabel"),
                "uuid": child.get("partuuid"),
                "fstype": fstype,
                "size_b": child.get("size") if isinstance(child.get("size"), int) else None,
                "start_lba": child.get("start"),
                "end_lba": child.get("end"),
                "content": content,
                "holders": child.get("holders"),
                "source": "lsblk",
            }
            if fstype in {"crypto_LUKS"}:
                partition["encryption"] = {
                    "encrypted": True,
                    "scheme": "luks",
                    "mapper_name": child.get("name"),
                    "mapped_device": child.get("path"),
                    "unlocked": True,
                }
            partitions.append({k: v for k, v in partition.items() if v is not None})
        return partitions

    def _populate_usage(self, metadata: dict[str, dict[str, Any]]) -> None:
        if not metadata:
            return

        partitions = psutil.disk_partitions(all=False)

        # On Linux, match partitions to drives by device name
        if platform.system().lower() == "linux":
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

        # On Windows, sum all partition usage for all drives
        elif platform.system().lower() == "windows":
            total_used = 0
            total_free = 0

            for part in partitions:
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    total_used += int(usage.used)
                    total_free += int(usage.free)
                except OSError:
                    self.logger.debug("Skipping usage for %s (unreadable).", part.mountpoint)
                    continue

            # Distribute total usage across all drives
            # This is an approximation since we can't easily map partitions to physical drives on Windows
            if metadata and (total_used > 0 or total_free > 0):
                # For simplicity, assign total usage to the first drive
                # In practice, Windows physical drive mapping is complex
                first_drive = next(iter(metadata.keys()))
                metadata[first_drive]["used_b"] = total_used
                metadata[first_drive]["available_b"] = total_free
                self.logger.debug("Assigned total usage to drive %s: %d used, %d free",
                                first_drive, total_used, total_free)

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

        lhm_info = self._collect_lhm_motherboard_info()
        if lhm_info:
            motherboard.update(lhm_info)

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
        currents: list[dict[str, Any]] = []
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
            elif category == "current":
                currents.append({"name": name, "current_a": value, "source": "lhm"})
            elif category == "power":
                powers.append({"name": name, "power_w": value, "source": "lhm"})

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
            if device.get("core_voltage_v") is not None:
                entry["core"]["voltage_v"] = device["core_voltage_v"]
            if device.get("core_power_w") is not None:
                entry["core"]["power_w"] = device["core_power_w"]
            if device.get("core_clock_hz") is not None:
                entry["core"]["clock_hz"] = device["core_clock_hz"]
            soc_entry: dict[str, Any] = {}
            if device.get("soc_voltage_v") is not None:
                soc_entry["voltage_v"] = device["soc_voltage_v"]
            if device.get("soc_power_w") is not None:
                soc_entry["power_w"] = device["soc_power_w"]
            if device.get("soc_clock_hz") is not None:
                soc_entry["clock_hz"] = device["soc_clock_hz"]
            if soc_entry:
                entry["soc"] = soc_entry
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
        if self._lhm_has_sensor_id(raw):
            return self._parse_lhm_tree(raw)

        motherboard_sensors: list[dict[str, Any]] = []
        gpu_sensors: dict[str, list[dict[str, Any]]] = {}
        battery_sensors: dict[str, list[dict[str, Any]]] = {}
        drive_sensors: dict[str, list[dict[str, Any]]] = {}
        gpus: list[dict[str, Any]] = []

        def walk(
            node: dict[str, Any],
            hardware_type: str | None,
            gpu_name: str | None,
            storage_name: str | None,
            battery_name: str | None,
        ) -> None:
            node_type = node.get("Type")
            node_text = node.get("Text") or ""
            next_hardware = hardware_type
            next_gpu_name = gpu_name
            next_storage_name = storage_name
            next_battery_name = battery_name
            if node_type == "Hardware":
                raw_hardware = (node.get("HardwareType") or "").lower()
                if raw_hardware.startswith("gpu"):
                    next_hardware = "gpu"
                elif raw_hardware in {"motherboard", "mainboard"}:
                    next_hardware = "motherboard"
                elif raw_hardware in {"hdd", "ssd", "storage", "nvme"}:
                    next_hardware = "storage"
                elif raw_hardware in {"battery"}:
                    next_hardware = "battery"
                else:
                    next_hardware = raw_hardware or hardware_type
                if next_hardware == "gpu":
                    gpus.append({"name": node_text})
                    gpu_sensors.setdefault(node_text, [])
                    next_gpu_name = node_text
                if next_hardware == "storage":
                    drive_sensors.setdefault(node_text, [])
                    next_storage_name = node_text
                if next_hardware == "battery":
                    battery_sensors.setdefault(node_text, [])
                    next_battery_name = node_text
                for child in node.get("Children", []):
                    walk(
                        child,
                        next_hardware,
                        next_gpu_name,
                        next_storage_name,
                        next_battery_name,
                    )
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
                if hardware_type == "storage":
                    if storage_name:
                        drive_sensors.setdefault(storage_name, []).append(sensor_entry)
                if hardware_type == "battery":
                    if battery_name:
                        battery_sensors.setdefault(battery_name, []).append(sensor_entry)
                return
            for child in node.get("Children", []):
                walk(
                    child,
                    next_hardware,
                    next_gpu_name,
                    next_storage_name,
                    next_battery_name,
                )

        for root in raw.get("Children", []):
            walk(root, None, None, None, None)

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

        parsed_batteries: list[dict[str, Any]] = []
        for name, sensors in battery_sensors.items():
            charge = next(
                (
                    sensor["value"]
                    for sensor in sensors
                    if sensor["category"] in {"charge", "level"}
                ),
                None,
            )
            voltage = next(
                (
                    sensor["value"]
                    for sensor in sensors
                    if sensor["category"] == "voltage"
                ),
                None,
            )
            current = next(
                (
                    sensor["value"]
                    for sensor in sensors
                    if sensor["category"] == "current"
                ),
                None,
            )
            power = next(
                (
                    sensor["value"]
                    for sensor in sensors
                    if sensor["category"] == "power"
                ),
                None,
            )
            if charge is None:
                continue
            discharging = False
            if power is not None:
                discharging = power < 0
            elif current is not None:
                discharging = current < 0
            entry: dict[str, Any] = {
                "name": name,
                "discharging": discharging,
                "charge_level_pct": float(charge),
            }
            if voltage is not None:
                entry["voltage_v"] = float(voltage)
            if current is not None:
                entry["current_a"] = float(current)
            if power is not None:
                entry["power_w"] = float(power)
            parsed_batteries.append(entry)

        parsed_drives: dict[str, dict[str, Any]] = {}
        for name, sensors in drive_sensors.items():
            temp = next(
                (
                    sensor["value"]
                    for sensor in sensors
                    if sensor["category"] == "temperature"
                ),
                None,
            )
            if temp is not None:
                parsed_drives[name] = {"temp_c": float(temp)}

        return {
            "motherboard_sensors": motherboard_sensors,
            "gpus": parsed_gpus,
            "batteries": parsed_batteries,
            "drives": parsed_drives,
        }

    def _collect_lhm_batteries(
        self, data: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not data:
            return []
        batteries = data.get("batteries", [])
        return [battery for battery in batteries if "charge_level_pct" in battery]

    def _collect_lhm_drives(
        self, data: dict[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        if not data:
            return {}
        return data.get("drives", {})

    def _collect_lhm_cpu(self) -> dict[str, Any] | None:
        data = self._read_lhm()
        if not data:
            return None
        return data.get("cpu", None)

    def _collect_lhm_motherboard_info(self) -> dict[str, Any] | None:
        data = self._read_lhm()
        if not data:
            return None
        return data.get("motherboard_info", None)

    def _lhm_has_sensor_id(self, raw: dict[str, Any]) -> bool:
        nodes = [raw]
        while nodes:
            node = nodes.pop()
            if isinstance(node, dict):
                if node.get("SensorId"):
                    return True
                nodes.extend(node.get("Children", []))
        return False

    def _parse_lhm_tree(self, raw: dict[str, Any]) -> dict[str, Any]:
        battery_map: dict[str, dict[str, Any]] = {}
        drive_map: dict[str, dict[str, Any]] = {}
        gpu_map: dict[str, dict[str, Any]] = {}

        def parse_numeric(value: str | None) -> float | None:
            if value is None:
                return None
            text = value.replace(",", "").strip()
            text = text.replace("/s", "").strip()
            for token in ["mWh", "°C", "GB", "MB", "KB", "V", "A", "W", "%"]:
                if token in text:
                    number = text.replace(token, "").strip()
                    try:
                        parsed = float(number)
                    except ValueError:
                        return None
                    if token == "GB":
                        return parsed * 1024 * 1024 * 1024
                    if token == "MB":
                        return parsed * 1024 * 1024
                    if token == "KB":
                        return parsed * 1024
                    return parsed
            try:
                return float(text)
            except ValueError:
                return None

        def classify_hardware(image_url: str) -> str | None:
            lowered = image_url.lower()
            if "battery" in lowered:
                return "battery"
            if "hdd" in lowered or "ssd" in lowered or "nvme" in lowered:
                return "storage"
            if "gpu" in lowered or "ati" in lowered or "nvidia" in lowered:
                return "gpu"
            if "mainboard" in lowered or "motherboard" in lowered:
                return "motherboard"
            return None

        cpu_entry: dict[str, Any] = {"cores": {}}
        motherboard_info: dict[str, Any] = {}
        motherboard_sensors: list[dict[str, Any]] = []

        def walk(node: dict[str, Any], hw_type: str | None, hw_name: str | None) -> None:
            image_url = node.get("ImageURL") or ""
            node_text = (node.get("Text") or "").replace("\x00", "").strip()
            detected = classify_hardware(image_url)
            next_type = detected or hw_type
            next_name = hw_name
            if detected in {"battery", "storage", "gpu", "motherboard", "cpu"}:
                next_name = node_text or hw_name

            if detected == "motherboard" and node_text:
                motherboard_info.setdefault("name", node_text)

            sensor_id = node.get("SensorId")
            sensor_type = (node.get("Type") or "").lower()
            if sensor_id or sensor_type:
                value = parse_numeric(node.get("Value"))
                if value is not None and next_type == "battery" and next_name:
                    battery = battery_map.setdefault(next_name, {})
                    if sensor_type == "voltage":
                        battery["voltage_v"] = float(value)
                    elif sensor_type == "current":
                        battery["current_a"] = float(value)
                    elif sensor_type == "power":
                        battery["power_w"] = float(value)
                    elif sensor_type == "level":
                        if "degradation" in node_text.lower():
                            battery["degradation_pct"] = float(value)
                        elif "charge" in node_text.lower():
                            battery["charge_level_pct"] = float(value)
                    elif sensor_type == "energy":
                        if "design" in node_text.lower():
                            battery["design_cap_mwh"] = float(value)
                        elif "fully" in node_text.lower():
                            battery["full_cap_mwh"] = float(value)
                        elif "remaining" in node_text.lower():
                            battery["remain_cap_mwh"] = float(value)
                if value is not None and next_type == "storage" and next_name:
                    drive = drive_map.setdefault(next_name, {})
                    label = node_text.lower()
                    if sensor_type == "temperature":
                        drive["temp_c"] = float(value)
                    elif sensor_type == "load":
                        if "read activity" in label:
                            drive["read_activity_pct"] = float(value)
                        elif "write activity" in label:
                            drive["write_activity_pct"] = float(value)
                    elif sensor_type == "throughput":
                        if "read rate" in label:
                            drive["read_rate_bps"] = int(value)
                        elif "write rate" in label:
                            drive["write_rate_bps"] = int(value)
                    elif sensor_type == "data":
                        if "data read" in label:
                            drive["total_read_b"] = int(value)
                        elif "data written" in label:
                            drive["total_write_b"] = int(value)
                if value is not None and next_type == "gpu" and next_name:
                    gpu = gpu_map.setdefault(next_name, {"engines": []})
                    label = node_text.lower()
                    if sensor_type == "temperature":
                        gpu["temp_c"] = float(value)
                    elif sensor_type == "load":
                        gpu["engines"].append(
                            {"name": node_text, "load": float(value)}
                        )
                    elif sensor_type == "voltage":
                        if "core" in label:
                            gpu["core_voltage_v"] = float(value)
                        elif "soc" in label:
                            gpu["soc_voltage_v"] = float(value)
                    elif sensor_type == "power":
                        if "core" in label:
                            gpu["core_power_w"] = float(value)
                        elif "soc" in label:
                            gpu["soc_power_w"] = float(value)
                    elif sensor_type == "clock":
                        if "core" in label:
                            gpu["core_clock_hz"] = float(value) * 1e6
                        elif "soc" in label:
                            gpu["soc_clock_hz"] = float(value) * 1e6
                        elif "memory" in label:
                            gpu["mem_clock_hz"] = float(value) * 1e6
                if value is not None and next_type == "cpu":
                    label = node_text.lower()
                    if sensor_type == "temperature" and "tctl" in label:
                        cpu_entry["temp_c"] = float(value)
                    elif sensor_type == "voltage":
                        if "core (svi2" in label:
                            cpu_entry["voltage_core_v"] = float(value)
                        elif "soc (svi2" in label:
                            cpu_entry["voltage_soc_v"] = float(value)
                        elif "vid" in label and "core #" in label:
                            core_index = self._parse_core_index(label)
                            if core_index is not None:
                                cpu_entry["cores"].setdefault(core_index, {})[
                                    "voltage_v"
                                ] = float(value)
                    elif sensor_type == "power" and "core #" in label:
                        core_index = self._parse_core_index(label)
                        if core_index is not None:
                            cpu_entry["cores"].setdefault(core_index, {})[
                                "power_w"
                            ] = float(value)
                    elif sensor_type == "clock" and "core #" in label:
                        core_index = self._parse_core_index(label)
                        if core_index is not None:
                            cpu_entry["cores"].setdefault(core_index, {})[
                                "clock_hz"
                            ] = float(value) * 1e6
                    elif sensor_type == "factor" and "core #" in label:
                        core_index = self._parse_core_index(label)
                        if core_index is not None:
                            cpu_entry["cores"].setdefault(core_index, {})[
                                "factor"
                            ] = float(value)
                if value is not None and next_type == "motherboard":
                    sensor_category = sensor_type
                    if sensor_category in {
                        "temperature",
                        "fan",
                        "voltage",
                        "power",
                        "current",
                    }:
                        motherboard_sensors.append(
                            {
                                "category": sensor_category,
                                "name": node_text,
                                "value": float(value),
                            }
                        )

            for child in node.get("Children", []):
                if isinstance(child, dict):
                    walk(child, next_type, next_name)

        walk(raw, None, None)

        batteries: list[dict[str, Any]] = []
        for name, data in battery_map.items():
            charge = data.get("charge_level_pct")
            if charge is None:
                continue
            discharging = False
            power = data.get("power_w")
            current = data.get("current_a")
            if power is not None:
                discharging = power < 0
            elif current is not None:
                discharging = current < 0
            entry = {
                "name": name,
                "discharging": discharging,
                "charge_level_pct": float(charge),
            }
            for key in [
                "voltage_v",
                "current_a",
                "power_w",
                "design_cap_mwh",
                "full_cap_mwh",
                "remain_cap_mwh",
                "degradation_pct",
            ]:
                if key in data:
                    entry[key] = data[key]
            batteries.append(entry)

        parsed_gpus: list[dict[str, Any]] = []
        for name, gpu_data in gpu_map.items():
            engines = gpu_data.get("engines", [])
            core_load = next(
                (
                    engine["load"]
                    for engine in engines
                    if "core" in engine["name"].lower()
                ),
                None,
            )
            gpu_entry: dict[str, Any] = {
                "name": name,
                "engines": [
                    {"name": engine["name"], "load": engine["load"]}
                    for engine in engines
                ],
                "core_load": core_load,
                "temp_c": gpu_data.get("temp_c"),
            }
            if gpu_data.get("core_voltage_v") is not None:
                gpu_entry["core_voltage_v"] = gpu_data["core_voltage_v"]
            if gpu_data.get("core_power_w") is not None:
                gpu_entry["core_power_w"] = gpu_data["core_power_w"]
            if gpu_data.get("core_clock_hz") is not None:
                gpu_entry["core_clock_hz"] = gpu_data["core_clock_hz"]
            if gpu_data.get("soc_voltage_v") is not None:
                gpu_entry["soc_voltage_v"] = gpu_data["soc_voltage_v"]
            if gpu_data.get("soc_power_w") is not None:
                gpu_entry["soc_power_w"] = gpu_data["soc_power_w"]
            if gpu_data.get("soc_clock_hz") is not None:
                gpu_entry["soc_clock_hz"] = gpu_data["soc_clock_hz"]
            if gpu_data.get("mem_clock_hz") is not None:
                gpu_entry["mem_clock_hz"] = gpu_data["mem_clock_hz"]
            parsed_gpus.append(gpu_entry)

        parsed_cpu = None
        if cpu_entry.get("temp_c") or cpu_entry.get("voltage_core_v") or cpu_entry["cores"]:
            parsed_cpu = cpu_entry

        return {
            "motherboard_sensors": motherboard_sensors,
            "gpus": parsed_gpus,
            "batteries": batteries,
            "drives": drive_map,
            "cpu": parsed_cpu,
            "motherboard_info": motherboard_info if motherboard_info else None,
        }
