from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
import re
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
class LoadAverage:
    """Exponentially weighted moving average for load calculation."""
    load_1m: float = 0.0
    load_5m: float = 0.0
    load_15m: float = 0.0
    last_update: float = 0.0

    def update(self, load_pct: float, timestamp: float) -> None:
        """Update load averages with a new sample.

        Uses exponential smoothing similar to Unix load average calculation.
        The smoothing factors approximate 1/e decay over the respective periods.
        """
        if self.last_update == 0:
            # First sample - initialize all averages
            self.load_1m = load_pct
            self.load_5m = load_pct
            self.load_15m = load_pct
            self.last_update = timestamp
            return

        elapsed = timestamp - self.last_update
        if elapsed <= 0:
            return

        # Smoothing factors based on elapsed time
        # exp(-elapsed/period) gives the weight for the old value
        alpha_1m = 1 - math.exp(-elapsed / 60)
        alpha_5m = 1 - math.exp(-elapsed / 300)
        alpha_15m = 1 - math.exp(-elapsed / 900)

        self.load_1m = alpha_1m * load_pct + (1 - alpha_1m) * self.load_1m
        self.load_5m = alpha_5m * load_pct + (1 - alpha_5m) * self.load_5m
        self.load_15m = alpha_15m * load_pct + (1 - alpha_15m) * self.load_15m
        self.last_update = timestamp


@dataclass
class MetricsState:
    last_disk: IoSnapshot | None = None
    last_net: NetSnapshot | None = None
    drive_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    load_avg: LoadAverage = field(default_factory=LoadAverage)


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
        # Add Intel iGPU if available and not already in gpus
        intel_gpu = self._collect_intel_gpu()
        if intel_gpu:
            if not gpus:
                gpus = []
            # Check if Intel GPU already reported via LHM
            intel_names = {"Intel", "UHD", "Iris", "HD Graphics"}
            has_intel = any(
                any(n in g.get("name", "") for n in intel_names) for g in gpus
            )
            if not has_intel:
                gpus.append(intel_gpu)
        if gpus:
            payload["gpus"] = gpus

        tpus = self._collect_tpus()
        if tpus:
            payload["tpus"] = tpus

        backup_providers = self._collect_backup_providers()
        if backup_providers:
            payload["backup_providers"] = backup_providers

        if self.config.enable_time_server:
            time_server = self._collect_time_server()
            if time_server:
                payload["time_server"] = time_server

        # OPNsense/Zenarmor integration (FreeBSD only)
        if self.config.enable_opnsense:
            opnsense = self._collect_opnsense()
            if opnsense:
                payload["opnsense"] = opnsense

            opnsense_plugins = self._collect_opnsense_plugins()
            if opnsense_plugins:
                payload["opnsense_plugins"] = opnsense_plugins

        if self.config.enable_zenarmor:
            zenarmor = self._collect_zenarmor()
            if zenarmor:
                payload["zenarmor"] = zenarmor

        self.logger.debug("Completed metrics payload collection.")
        return payload

    def _collect_host(self, ts: str) -> dict[str, Any]:
        boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
        uptime_s = int(datetime.now(timezone.utc).timestamp() - boot.timestamp())
        host: dict[str, Any] = {
            "name": socket.gethostname(),
            "system": platform.system(),
            "os": platform.platform(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "boot_time": boot.isoformat(),
            "uptime_s": uptime_s,
            "process_count": len(psutil.pids()),
        }

        # Add system stats (context switches, interrupts) if available
        try:
            cpu_stats = psutil.cpu_stats()
            host["context_switches"] = cpu_stats.ctx_switches
            host["interrupts"] = cpu_stats.interrupts
        except Exception:
            self.logger.debug("Failed to collect CPU stats.")

        # Add device model and serial from device-tree (Linux SBCs like Raspberry Pi)
        if platform.system().lower() == "linux":
            model = self._read_file("/proc/device-tree/model")
            if model:
                host["model"] = model.rstrip("\x00").strip()
            serial = self._read_file("/sys/firmware/devicetree/base/serial-number")
            if serial:
                host["serial"] = serial.rstrip("\x00").strip()

            # Collect Raspberry Pi / SBC specific metrics via vcgencmd
            throttling = self._collect_vcgencmd_throttling()
            if throttling:
                host["throttling"] = throttling

            sbc = self._collect_vcgencmd_sbc()
            if sbc:
                host["sbc"] = sbc

        # Add device model, serial, and chassis type from WMI (Windows)
        elif platform.system().lower() == "windows":
            hw_info = self._collect_host_hardware_windows()
            if hw_info:
                host.update(hw_info)

        return host

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
        summary = "OK" if not issues else f"Issues: {'; '.join(issues)}"
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

        updates = self._collect_updates()
        if updates:
            health["updates"] = updates

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

        # Get or compute load averages
        current_load = cpu_entry["load_pct"]
        now = datetime.now(timezone.utc).timestamp()

        if hasattr(os, "getloadavg"):
            # Unix - use native load averages
            load_1m, load_5m, load_15m = os.getloadavg()
            cpu_entry.update(
                {
                    "load_1m": float(load_1m),
                    "load_5m": float(load_5m),
                    "load_15m": float(load_15m),
                }
            )
        else:
            # Windows - compute load averages from CPU usage samples
            self.state.load_avg.update(current_load, now)
            # Only include if we have meaningful data (not first sample)
            if self.state.load_avg.last_update > 0:
                cpu_entry.update(
                    {
                        "load_1m": round(self.state.load_avg.load_1m, 2),
                        "load_5m": round(self.state.load_avg.load_5m, 2),
                        "load_15m": round(self.state.load_avg.load_15m, 2),
                    }
                )

        # Add CPU frequency data per-core
        try:
            freq_per_cpu = psutil.cpu_freq(percpu=True)
            if freq_per_cpu and len(freq_per_cpu) == len(cores):
                for i, freq in enumerate(freq_per_cpu):
                    if freq and freq.current:
                        # Convert MHz to Hz
                        cores[i]["clock_hz"] = int(freq.current * 1_000_000)
            elif freq_per_cpu:
                # Fallback: single frequency for all cores
                freq = freq_per_cpu[0] if freq_per_cpu else psutil.cpu_freq()
                if freq and freq.current:
                    for core in cores:
                        core["clock_hz"] = int(freq.current * 1_000_000)
        except Exception:
            self.logger.debug("Failed to collect CPU frequency data.")

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
            for key in ["temp_c", "voltage_core_v", "voltage_soc_v", "power_w"]:
                if key in lhm_cpu and lhm_cpu[key] is not None:
                    cpu_entry[key] = lhm_cpu[key]
            core_metrics = lhm_cpu.get("cores", {})
            for core in cores:
                index = int(core["name"].split()[-1])
                lhm_index = index + 1
                if lhm_index in core_metrics:
                    core.update(core_metrics[lhm_index])

        # Add CPU frequency scaling info (Linux sysfs)
        if platform.system().lower() == "linux":
            freq_scaling = self._collect_cpu_freq_scaling()
            if freq_scaling:
                cpu_entry.update(freq_scaling)

        # Add FreeBSD CPU temperatures via sysctl
        if platform.system().lower() == "freebsd":
            freebsd_temps = self._collect_cpu_temps_freebsd()
            if freebsd_temps:
                # Set overall CPU temp from first core
                if 0 in freebsd_temps and "temp_c" not in cpu_entry:
                    cpu_entry["temp_c"] = freebsd_temps[0]
                # Add per-core temps
                for core in cores:
                    core_idx = int(core["name"].split()[-1])
                    if core_idx in freebsd_temps:
                        core["temp_c"] = freebsd_temps[core_idx]

        return [cpu_entry]

    def _collect_memory(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        system_mem: dict[str, Any] = {
            "used_b": int(vm.used),
            "available_b": int(vm.available),
            "load_pct": float(vm.percent),
            "total_b": int(vm.total),
        }

        # Add detailed memory breakdown on Linux (from psutil or /proc/meminfo)
        if platform.system().lower() == "linux":
            # psutil provides these on Linux
            if hasattr(vm, "buffers") and vm.buffers:
                system_mem["buffers_b"] = int(vm.buffers)
            if hasattr(vm, "cached") and vm.cached:
                system_mem["cached_b"] = int(vm.cached)
            if hasattr(vm, "shared") and vm.shared:
                system_mem["shmem_b"] = int(vm.shared)

            # Get additional details from /proc/meminfo
            meminfo = self._parse_meminfo()
            if meminfo:
                if "Dirty" in meminfo:
                    system_mem["dirty_b"] = meminfo["Dirty"]
                if "Slab" in meminfo:
                    system_mem["slab_b"] = meminfo["Slab"]

        memory: dict[str, Any] = {"system": system_mem}

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
        is_windows = platform.system().lower() == "windows"
        is_linux = platform.system().lower() == "linux"

        # Build partition-to-drive mapping
        partition_map: dict[str, str] = {}
        for drive_name, meta in drive_metadata.items():
            for partition in meta.get("partitions", []):
                partition_name = partition.get("name")
                if partition_name:
                    partition_map[partition_name] = drive_name

        # On Linux, build device-to-label/uuid mapping from lsblk
        device_metadata: dict[str, dict[str, Any]] = {}
        if is_linux:
            device_metadata = self._collect_device_metadata_lsblk()

        # Windows-specific: map drive letters to physical disk numbers
        windows_partition_map: dict[str, int] = {}
        if is_windows:
            windows_partition_map = self._windows_partition_map()

        # Get LHM drive data for metrics
        lhm_data = self._read_lhm()
        lhm_drives = self._collect_lhm_drives(lhm_data)

        # Build a mapping from disk number to LHM drive data (for Windows)
        # When there's only one drive in each, they match
        lhm_by_disk: dict[int, dict[str, Any]] = {}
        if is_windows and lhm_drives:
            if len(drive_metadata) == 1 and len(lhm_drives) == 1:
                # Single drive case - map disk 0 to the LHM data
                lhm_by_disk[0] = next(iter(lhm_drives.values()))
            # TODO: Multi-drive matching could be enhanced

        # Get psutil disk I/O counters for computing rates on Linux
        io_stats = psutil.disk_io_counters(perdisk=True)
        now = datetime.now(timezone.utc).timestamp()
        drive_io_rates: dict[str, dict[str, Any]] = {}
        if is_linux and self.state.last_disk:
            elapsed = now - self.state.last_disk.timestamp
            if elapsed > 0:
                for disk_name, io_entry in io_stats.items():
                    if disk_name in self.state.last_disk.values:
                        prev = self.state.last_disk.values[disk_name]
                        drive_io_rates[disk_name] = {
                            "read_rate_bps": int((io_entry.read_bytes - prev.read_bytes) / elapsed),
                            "write_rate_bps": int((io_entry.write_bytes - prev.write_bytes) / elapsed),
                        }
                        # Calculate activity from busy_time
                        if hasattr(io_entry, "busy_time") and hasattr(prev, "busy_time"):
                            busy_delta_ms = io_entry.busy_time - prev.busy_time
                            elapsed_ms = elapsed * 1000
                            if elapsed_ms > 0:
                                activity_pct = min(100.0, (busy_delta_ms / elapsed_ms) * 100)
                                drive_io_rates[disk_name]["read_activity_pct"] = round(activity_pct, 1)
                                drive_io_rates[disk_name]["write_activity_pct"] = round(activity_pct, 1)

        # Collect partitions first
        partitions = psutil.disk_partitions(all=False)

        # Get BitLocker status on Windows
        bitlocker_status: dict[str, dict[str, Any]] = {}
        volume_metadata: dict[str, dict[str, Any]] = {}
        if is_windows:
            mountpoints = [p.mountpoint for p in partitions]
            bitlocker_status = self._windows_bitlocker_status(mountpoints)
            volume_metadata = self._windows_volume_metadata()

        for part in partitions:
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                self.logger.debug("Skipping filesystem at %s (unreadable).", part.mountpoint)
                continue
            entry: dict[str, Any] = {
                "name": part.device,
                "mountpoint": part.mountpoint,
                "format": part.fstype,
                "used_b": int(usage.used),
                "available_b": int(usage.free),
            }

            # Determine backing drive
            drive_name = partition_map.get(part.device)
            disk_number: int | None = None

            if drive_name:
                # Linux path - have partition info from lsblk
                entry["backing_blockdev"] = {
                    "device": part.device,
                    "drive_name": drive_name,
                    "partition_name": part.device,
                }
            elif is_windows:
                # Windows path - use partition map
                disk_number = windows_partition_map.get(part.mountpoint)
                if disk_number is not None:
                    drive_name = f"PhysicalDrive{disk_number}"
                    entry["backing_blockdev"] = {
                        "device": part.device,
                        "drive_name": drive_name,
                    }

            # Add I/O metrics - prefer LHM, fall back to psutil-computed rates
            io_source = None
            if drive_name and drive_name in lhm_drives:
                io_source = lhm_drives[drive_name]
            elif disk_number is not None and disk_number in lhm_by_disk:
                io_source = lhm_by_disk[disk_number]
            elif drive_name and drive_name in drive_io_rates:
                # Fall back to psutil-computed rates on Linux
                io_source = drive_io_rates[drive_name]

            if io_source:
                if "read_rate_bps" in io_source:
                    entry["read_rate_bps"] = io_source["read_rate_bps"]
                if "write_rate_bps" in io_source:
                    entry["write_rate_bps"] = io_source["write_rate_bps"]
                if "read_activity_pct" in io_source:
                    entry["read_activity_pct"] = io_source["read_activity_pct"]
                if "write_activity_pct" in io_source:
                    entry["write_activity_pct"] = io_source["write_activity_pct"]

            # Add encryption info if available
            if part.mountpoint in bitlocker_status:
                entry["encryption"] = bitlocker_status[part.mountpoint]

            # Add volume metadata (label, uuid, fstype) on Windows
            if part.mountpoint in volume_metadata:
                vol_meta = volume_metadata[part.mountpoint]
                if "label" in vol_meta:
                    entry["label"] = vol_meta["label"]
                if "uuid" in vol_meta:
                    entry["uuid"] = vol_meta["uuid"]
                # Update fstype if available from Get-Volume (more reliable on Windows)
                if "fstype" in vol_meta:
                    entry["format"] = vol_meta["fstype"]

            # Add label/uuid from lsblk on Linux
            if is_linux and part.device in device_metadata:
                dev_meta = device_metadata[part.device]
                if "label" in dev_meta and dev_meta["label"]:
                    entry["label"] = dev_meta["label"]
                if "uuid" in dev_meta and dev_meta["uuid"]:
                    entry["uuid"] = dev_meta["uuid"]

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
                "manufacturer": (meta.get("manufacturer") or "unknown").strip(),
                "model": (meta.get("model") or "unknown").strip(),
                "used_b": meta.get("used_b", 0),
                "available_b": meta.get("available_b", 0),
                "total_read_b": int(io_entry.read_bytes),
                "total_write_b": int(io_entry.write_bytes),
            }
            # Only include optional fields if they have values
            if meta.get("listed_cap_b") is not None:
                entry["listed_cap_b"] = meta["listed_cap_b"]
            if meta.get("serial"):
                entry["serial_number"] = meta["serial"]
            if meta.get("firmware"):
                entry["firmware_version"] = meta["firmware"]
            if meta.get("partitions"):
                entry["partitions"] = meta["partitions"]

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
                    # Calculate activity percentage from busy_time (ms)
                    if hasattr(io_entry, "busy_time") and hasattr(prev, "busy_time"):
                        busy_delta_ms = io_entry.busy_time - prev.busy_time
                        elapsed_ms = elapsed * 1000
                        if elapsed_ms > 0:
                            activity_pct = min(100.0, (busy_delta_ms / elapsed_ms) * 100)
                            entry["read_activity_pct"] = round(activity_pct, 1)
                            entry["write_activity_pct"] = round(activity_pct, 1)

            smart = meta.get("smart")
            if smart:
                entry["smart"] = smart

            # Try to find matching LHM drive data
            lhm_match = None
            if name in lhm_drives:
                # Direct name match (Linux/lsblk)
                lhm_match = lhm_drives[name]
            elif len(lhm_drives) == 1 and len(metadata) == 1:
                # Single drive on both sides - match them (Windows)
                lhm_name, lhm_match = next(iter(lhm_drives.items()))
                # Update model from LHM since Windows metadata lacks it
                entry["model"] = lhm_name

            if lhm_match:
                for key in [
                    "temp_c",
                    "read_activity_pct",
                    "write_activity_pct",
                    "read_rate_bps",
                    "write_rate_bps",
                    "total_read_b",
                    "total_write_b",
                ]:
                    if key in lhm_match and lhm_match[key] is not None:
                        entry[key] = lhm_match[key]

            # Get drive temp from Linux sensors if not already set
            if "temp_c" not in entry and platform.system().lower() == "linux":
                drive_temp = self._get_drive_temp_from_sensors(name)
                if drive_temp is not None:
                    entry["temp_c"] = drive_temp

            drives.append(entry)

        self.state.last_disk = IoSnapshot(timestamp=now, values=io_stats)
        return drives

    def _collect_ifaces(self) -> list[dict[str, Any]]:
        addrs = psutil.net_if_addrs()
        io_stats = psutil.net_io_counters(pernic=True)
        # net_if_stats() can fail with OSError in containers without proper network support
        try:
            iface_stats = psutil.net_if_stats()
        except OSError:
            self.logger.debug("Failed to get network interface stats (ioctl not supported).")
            iface_stats = {}
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

            entry: dict[str, Any] = {
                "name": iface,
                "ipv4": ipv4,
                "ipv6": ipv6,
            }

            # Include MAC address only if available
            if mac is not None:
                entry["mac"] = mac

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

            # Add link details based on platform
            if platform.system().lower() == "linux":
                link_details = self._collect_iface_link_details(iface)
                if link_details:
                    entry.update(link_details)

                # Add WiFi metrics for wireless interfaces
                wifi_metrics = self._collect_wifi_metrics(iface)
                if wifi_metrics:
                    entry["wifi"] = wifi_metrics
            elif platform.system().lower() == "windows":
                link_details = self._collect_iface_link_details_windows(iface)
                if link_details:
                    entry.update(link_details)

                # Add WiFi metrics for wireless interfaces
                wifi_metrics = self._collect_wifi_metrics_windows(iface)
                if wifi_metrics:
                    entry["wifi"] = wifi_metrics

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
            # Skip read-only squashfs mounts (snap packages) - they're always 100% used
            if part.fstype == "squashfs":
                continue
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
        if platform.system().lower() == "windows":
            return self._collect_services_windows()
        if platform.system().lower() == "freebsd":
            return self._collect_services_freebsd()
        if platform.system().lower() != "linux":
            self.logger.debug("Service checks only supported on Linux/systemd, FreeBSD, and Windows.")
            return []
        services: list[dict[str, Any]] = []
        for service in self.health.services:
            output = self._run_command(
                [
                    self.config.systemctl_path,
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
            entry: dict[str, Any] = {
                "name": service,
                "ok": ok,
                "status": status,
                "loaded": loaded,
            }
            if not ok:
                entry["detail"] = f"{active}/{sub}"
            services.append(entry)
        return services

    def _collect_services_windows(self) -> list[dict[str, Any]]:
        """Collect Windows service status."""
        if not self.health.services:
            return []

        # Query all requested services in one call
        service_names = ",".join(f'"{s}"' for s in self.health.services)
        result = self._run_command(
            ["powershell", "-Command",
             f"Get-Service -Name {service_names} -ErrorAction SilentlyContinue | "
             "Select-Object Name, Status, StartType | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )

        services: list[dict[str, Any]] = []
        found_services: set[str] = set()

        if result:
            try:
                data = json.loads(result)
                if isinstance(data, dict):
                    data = [data]
                for svc in data:
                    name = svc.get("Name")
                    if not name:
                        continue
                    found_services.add(name.lower())
                    # Status: 1=Stopped, 2=StartPending, 3=StopPending, 4=Running,
                    # 5=ContinuePending, 6=PausePending, 7=Paused
                    status_code = svc.get("Status", 0)
                    status_map = {
                        1: "inactive",
                        2: "activating",
                        3: "deactivating",
                        4: "active",
                        5: "activating",
                        6: "deactivating",
                        7: "inactive",
                    }
                    status = status_map.get(status_code, "failed")
                    ok = status_code == 4  # Running
                    entry: dict[str, Any] = {
                        "name": name,
                        "ok": ok,
                        "status": status,
                    }
                    if not ok:
                        entry["detail"] = f"Status code: {status_code}"
                    # StartType: 0=Boot, 1=System, 2=Automatic, 3=Manual, 4=Disabled
                    start_type = svc.get("StartType", 0)
                    if start_type == 4:
                        entry["loaded"] = "masked"
                    else:
                        entry["loaded"] = "loaded"
                    services.append(entry)
            except json.JSONDecodeError:
                self.logger.debug("Failed to parse Windows service data.")

        # Add entries for services that weren't found
        for service in self.health.services:
            if service.lower() not in found_services:
                services.append({
                    "name": service,
                    "ok": False,
                    "status": "failed",
                    "detail": "Service not found",
                    "loaded": "not-found",
                })

        return services

    def _collect_updates(self) -> dict[str, Any] | None:
        """Collect pending software updates."""
        if platform.system().lower() == "windows":
            return self._collect_updates_windows()
        elif platform.system().lower() == "linux":
            return self._collect_updates_linux()
        elif platform.system().lower() == "freebsd":
            return self._collect_updates_freebsd()
        return None

    def _collect_updates_linux(self) -> dict[str, Any] | None:
        """Collect pending Linux updates using apt or dnf."""
        # Try apt list --upgradable (Debian/Ubuntu)
        output = self._run_command(
            [self.config.apt_path, "list", "--upgradable"], stderr=subprocess.DEVNULL
        )
        if output is not None:
            lines = [l for l in output.strip().split("\n") if l and not l.startswith("Listing")]
            if lines:
                return {
                    "pending": len(lines),
                    "source": "apt",
                }
            return {"pending": 0, "source": "apt"}

        # Try dnf check-update (Fedora/RHEL)
        output = self._run_command(
            [self.config.dnf_path, "check-update", "-q"], stderr=subprocess.DEVNULL
        )
        if output is not None:
            lines = [l for l in output.strip().split("\n") if l and not l.startswith(" ")]
            return {
                "pending": len(lines),
                "source": "dnf",
            }

        return None

    def _collect_updates_windows(self) -> dict[str, Any] | None:
        """Collect pending Windows Updates using COM API."""
        result = self._run_command(
            ["powershell", "-Command",
             "$UpdateSession = New-Object -ComObject Microsoft.Update.Session; "
             "$UpdateSearcher = $UpdateSession.CreateUpdateSearcher(); "
             "try { $Updates = $UpdateSearcher.Search('IsInstalled=0').Updates; "
             "$Updates | Select-Object Title, @{N='KB';E={($_.KBArticleIDs | Select-Object -First 1)}} "
             "| ConvertTo-Json -Compress } catch { '[]' }"],
            stderr=subprocess.DEVNULL,
        )
        if not result:
            return None

        try:
            data = json.loads(result)
            if isinstance(data, dict):
                data = [data]
            if not data:
                return None

            items: list[dict[str, Any]] = []
            for update in data:
                title = update.get("Title")
                if title:
                    entry: dict[str, Any] = {"name": title}
                    kb = update.get("KB")
                    if kb:
                        entry["version"] = f"KB{kb}"
                    items.append(entry)

            return {
                "source": "windows_update",
                "pending": len(items),
                "items": items[:20],  # Limit to 20 items to avoid huge payloads
            }
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse Windows Update data.")
            return None

    def _collect_services_freebsd(self) -> list[dict[str, Any]]:
        """Collect FreeBSD service status using service or pluginctl command.

        On OPNsense, many services are managed via pluginctl rather than the
        standard FreeBSD service command. When enable_opnsense is true, we try
        pluginctl first, then fall back to the service command.
        """
        if not self.health.services:
            return []

        services: list[dict[str, Any]] = []
        for service_name in self.health.services:
            is_running = False
            checked_via = "service"

            # On OPNsense, try pluginctl first for service status
            if self.config.enable_opnsense:
                try:
                    result = subprocess.run(
                        [self.config.pluginctl_path, service_name, "status"],
                        capture_output=True,
                        text=True,
                    )
                    # pluginctl returns 0 if service is running
                    if result.returncode == 0:
                        is_running = True
                        checked_via = "pluginctl"
                    elif "unknown" not in result.stderr.lower():
                        # pluginctl recognized the service but it's not running
                        checked_via = "pluginctl"
                except FileNotFoundError:
                    pass  # pluginctl not available, fall through to service

            # Fall back to standard service command if pluginctl didn't confirm running
            if not is_running and checked_via == "service":
                try:
                    result = subprocess.run(
                        [self.config.service_path, service_name, "status"],
                        capture_output=True,
                        text=True,
                    )
                    is_running = result.returncode == 0
                except FileNotFoundError:
                    pass

            status = "active" if is_running else "inactive"

            entry: dict[str, Any] = {
                "name": service_name,
                "ok": is_running,
                "status": status,
                "loaded": "loaded",  # FreeBSD doesn't have systemd's loaded state
            }
            if not is_running:
                entry["detail"] = f"checked via {checked_via}"
            services.append(entry)

        return services

    def _collect_updates_freebsd(self) -> dict[str, Any] | None:
        """Collect pending FreeBSD updates using pkg."""
        # Check for available package updates
        output = self._run_command(
            [self.config.pkg_path, "version", "-vRL="],
            stderr=subprocess.DEVNULL
        )

        updates: dict[str, Any] = {"source": "pkg"}
        pending_items: list[dict[str, str]] = []
        security_issues = 0

        if output is not None:
            # Parse pkg version output - lines with '<' indicate available updates
            for line in output.strip().split("\n"):
                if line and "<" in line:
                    parts = line.split()
                    if parts:
                        pkg_name = parts[0]
                        pending_items.append({"name": pkg_name})

            updates["pending"] = len(pending_items)
            if pending_items:
                updates["items"] = pending_items[:20]  # Limit to 20 items

        # Check for security vulnerabilities with pkg audit
        audit_output = self._run_command(
            [self.config.pkg_path, "audit", "-F"],
            stderr=subprocess.DEVNULL
        )
        if audit_output is not None:
            # Count lines that contain vulnerability info
            vuln_lines = [l for l in audit_output.strip().split("\n")
                         if l and "is vulnerable" in l.lower()]
            security_issues = len(vuln_lines)

        if security_issues > 0:
            updates["security_issues"] = security_issues

        if not output and not audit_output:
            return None

        if "pending" not in updates:
            updates["pending"] = 0

        return updates

    def _is_freebsd(self) -> bool:
        """Check if running on FreeBSD."""
        return platform.system().lower() == "freebsd"

    def _is_opnsense(self) -> bool:
        """Check if running on OPNsense (FreeBSD-based firewall)."""
        if not self._is_freebsd():
            return False
        # Check for opnsense-version command
        result = subprocess.run(
            ["opnsense-version", "-v"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def _collect_opnsense(self) -> dict[str, Any] | None:
        """Collect OPNsense system information."""
        if not self.config.enable_opnsense or not self._is_freebsd():
            return None

        opnsense: dict[str, Any] = {}

        # Get OPNsense version
        version_output = self._run_command(["opnsense-version", "-v"])
        if version_output:
            opnsense["version"] = version_output.strip()

        # Get system status via configctl
        status_output = self._run_command(
            [self.config.configctl_path, "system", "status"]
        )
        if status_output:
            opnsense["status_raw"] = status_output.strip()

        return opnsense if opnsense else None

    def _collect_opnsense_plugins(self) -> list[dict[str, Any]] | None:
        """Collect installed OPNsense plugins.

        Uses pkg query to list installed packages matching the os-* pattern,
        which are OPNsense plugins.
        """
        if not self.config.enable_opnsense or not self._is_freebsd():
            return None

        # List all installed os-* packages (OPNsense plugins)
        # pkg query format: %n = name, %v = version
        output = self._run_command(
            [self.config.pkg_path, "query", "-g", "%n %v", "os-*"]
        )
        if not output:
            return None

        plugins: list[dict[str, Any]] = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = line.split(None, 1)  # Split on first whitespace
            if parts:
                plugin: dict[str, Any] = {"name": parts[0]}
                if len(parts) > 1:
                    plugin["version"] = parts[1]
                plugins.append(plugin)

        return plugins if plugins else None

    def _collect_zenarmor(self) -> dict[str, Any] | None:
        """Collect Zenarmor (Sensei) service status.

        Based on actual CLI testing, zenarmorctl only exposes service status:
        - zenarmorctl cloud status -> "senpai is running as pid XXXX"
        - zenarmorctl engine status -> "eastpect is running as pid XXXX"
        """
        if not self.config.enable_zenarmor or not self._is_freebsd():
            return None

        zenarmor: dict[str, Any] = {}

        # Check cloud connector (senpai) status
        cloud_output = self._run_command(
            [self.config.zenarmorctl_path, "cloud", "status"]
        )
        if cloud_output:
            # Parse "senpai is running as pid XXXX" or "senpai is not running"
            cloud_lower = cloud_output.lower()
            zenarmor["cloud_running"] = "is running" in cloud_lower
            # Extract PID if running
            if zenarmor["cloud_running"] and "pid" in cloud_lower:
                try:
                    pid_str = cloud_output.split("pid")[-1].strip()
                    zenarmor["cloud_pid"] = int(pid_str.split()[0])
                except (ValueError, IndexError):
                    pass
        else:
            zenarmor["cloud_running"] = False

        # Check inspection engine (eastpect) status
        engine_output = self._run_command(
            [self.config.zenarmorctl_path, "engine", "status"]
        )
        if engine_output:
            # Parse "eastpect is running as pid XXXX" or "eastpect is not running"
            engine_lower = engine_output.lower()
            zenarmor["engine_running"] = "is running" in engine_lower
            # Extract PID if running
            if zenarmor["engine_running"] and "pid" in engine_lower:
                try:
                    pid_str = engine_output.split("pid")[-1].strip()
                    zenarmor["engine_pid"] = int(pid_str.split()[0])
                except (ValueError, IndexError):
                    pass
        else:
            zenarmor["engine_running"] = False

        return zenarmor if zenarmor else None

    def _collect_cpu_temps_freebsd(self) -> dict[int, float]:
        """Collect CPU core temperatures on FreeBSD via sysctl.

        Returns a dict mapping core index to temperature in Celsius.
        Uses dev.cpu.N.temperature if available, falls back to
        hw.acpi.thermal.tz0 (ACPI thermal zone) as package temp.
        """
        temps: dict[int, float] = {}

        output = self._run_command(
            [self.config.sysctl_path, "-a"],
            stderr=subprocess.DEVNULL
        )
        if not output:
            return temps

        for line in output.split("\n"):
            # Look for dev.cpu.N.temperature: XXX.XC
            if "dev.cpu." in line and ".temperature:" in line:
                try:
                    # Parse "dev.cpu.0.temperature: 45.0C"
                    key, value = line.split(":", 1)
                    core_num = int(key.split(".")[2])
                    temp_str = value.strip().rstrip("CK")
                    temp_val = float(temp_str)
                    # Convert from Kelvin if > 200 (likely Kelvin)
                    if temp_val > 200:
                        temp_val = temp_val - 273.15
                    temps[core_num] = temp_val
                except (ValueError, IndexError):
                    continue

        # Fallback: use ACPI thermal zone tz0 as CPU package temp if no per-core temps
        if not temps:
            for line in output.split("\n"):
                # Look for hw.acpi.thermal.tz0.temperature: XXX.XC
                if "hw.acpi.thermal.tz0.temperature:" in line:
                    try:
                        _, value = line.split(":", 1)
                        temp_str = value.strip().rstrip("CK")
                        temp_val = float(temp_str)
                        if temp_val > 200:
                            temp_val = temp_val - 273.15
                        temps[0] = temp_val
                        self.logger.debug(
                            "Using ACPI thermal zone tz0 (%.1fC) as CPU temp", temp_val
                        )
                    except (ValueError, IndexError):
                        continue
                    break

        return temps

    def _collect_thermal_zones_freebsd(self) -> list[dict[str, Any]]:
        """Collect thermal zone temperatures on FreeBSD via sysctl.

        Returns a list of thermal zone entries with name and temp_c.
        """
        zones: list[dict[str, Any]] = []

        output = self._run_command(
            [self.config.sysctl_path, "-a"],
            stderr=subprocess.DEVNULL
        )
        if not output:
            return zones

        for line in output.split("\n"):
            # Look for hw.acpi.thermal.tzN.temperature
            if "hw.acpi.thermal." in line and ".temperature:" in line:
                try:
                    key, value = line.split(":", 1)
                    # Extract zone name like "tz0"
                    parts = key.split(".")
                    zone_name = parts[3] if len(parts) > 3 else "unknown"
                    temp_str = value.strip().rstrip("CK")
                    temp_val = float(temp_str)
                    # Convert from Kelvin if > 200
                    if temp_val > 200:
                        temp_val = temp_val - 273.15
                    zones.append({
                        "name": zone_name,
                        "temp_c": temp_val,
                    })
                except (ValueError, IndexError):
                    continue

        return zones

    def _collect_freebsd_sensors(self) -> dict[str, Any] | None:
        """Collect sensor data on FreeBSD.

        Returns motherboard-compatible dict with temps from thermal zones.
        """
        if not self._is_freebsd():
            return None

        temps: list[dict[str, Any]] = []

        # Collect thermal zone temperatures
        zones = self._collect_thermal_zones_freebsd()
        for zone in zones:
            temps.append({
                "name": zone["name"],
                "temp_c": zone["temp_c"],
                "source": "sysctl",
            })

        if not temps:
            return None

        return {"temps": temps}

    def _collect_hassio_addons(self) -> list[dict[str, Any]]:
        """Collect addon status using Home Assistant Supervisor API."""
        if not self.health.containers:
            return []

        # Get Supervisor token from environment (set by HassOS for addons with hassio_api: true)
        supervisor_token = os.environ.get("SUPERVISOR_TOKEN")
        if not supervisor_token:
            self.logger.warning("SUPERVISOR_TOKEN not set, cannot query addon status")
            return [
                {
                    "name": slug,
                    "ok": False,
                    "status": "unknown",
                    "detail": "no supervisor token",
                }
                for slug in self.health.containers
            ]

        self.logger.debug("Using Supervisor API to query %d addons", len(self.health.containers))
        addons: list[dict[str, Any]] = []
        for slug in self.health.containers:
            try:
                from urllib.request import Request
                url = f"http://supervisor/addons/{slug}/info"
                req = Request(url)
                req.add_header("Authorization", f"Bearer {supervisor_token}")
                with urlopen(req, timeout=5) as response:
                    info = json.loads(response.read().decode("utf-8"))

                data = info.get("data", info)
                state = data.get("state", "unknown")
                name = data.get("name", slug)
                version = data.get("version")

                # HassOS states: started, stopped, unknown
                if state == "started":
                    status = "running"
                    ok = True
                elif state == "stopped":
                    status = "stopped"
                    ok = False
                else:
                    status = state
                    ok = False

                addon_info: dict[str, Any] = {
                    "name": name,
                    "ok": ok,
                    "status": status,
                    "detail": state,
                }
                if version:
                    addon_info["version"] = version
                addons.append(addon_info)
                self.logger.debug("Addon %s: state=%s", slug, state)

            except Exception as e:
                self.logger.warning("Failed to get addon info for %s: %s", slug, e)
                addons.append({
                    "name": slug,
                    "ok": False,
                    "status": "unknown",
                    "detail": str(e),
                })

        return addons

    def _collect_containers(self) -> list[dict[str, Any]]:
        if not self.health.containers:
            return []

        # Use HassOS CLI if enabled
        if self.health.enable_hassio:
            return self._collect_hassio_addons()

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
        output = self._run_command([self.config.lsblk_path, "-J", "-b", "-O"])
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
                "manufacturer": (block.get("vendor") or "unknown").strip(),
                "model": (block.get("model") or "unknown").strip(),
                "type": drive_type,
                "listed_cap_b": block.get("size")
                if isinstance(block.get("size"), int)
                else None,
                "serial": (block.get("serial") or "").strip() or None,
                "firmware": (block.get("rev") or "").strip() or None,
                "partitions": partitions,
            }
        return metadata

    def _collect_device_metadata_lsblk(self) -> dict[str, dict[str, Any]]:
        """Collect device label/uuid metadata from lsblk for filesystems."""
        if platform.system().lower() != "linux":
            return {}
        output = self._run_command([self.config.lsblk_path, "-J", "-b", "-o", "NAME,PATH,LABEL,UUID,FSTYPE"])
        if output is None:
            return {}
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return {}

        device_meta: dict[str, dict[str, Any]] = {}

        def traverse(devices: list[dict[str, Any]]) -> None:
            for dev in devices:
                path = dev.get("path")
                if path:
                    meta: dict[str, Any] = {}
                    if dev.get("label"):
                        meta["label"] = dev["label"]
                    if dev.get("uuid"):
                        meta["uuid"] = dev["uuid"]
                    if meta:
                        device_meta[path] = meta
                # Recurse into children (partitions, LVM, crypt, etc.)
                if "children" in dev:
                    traverse(dev["children"])

        traverse(data.get("blockdevices", []))
        return device_meta

    def _drive_metadata_windows(self) -> dict[str, dict[str, Any]]:
        """Create drive metadata on Windows from WMI and PowerShell."""
        io_stats = psutil.disk_io_counters(perdisk=True)
        if not io_stats:
            return {}

        metadata: dict[str, dict[str, Any]] = {}

        # Get detailed drive info from WMI
        wmi_drives: dict[int, dict[str, Any]] = {}
        result = self._run_command(
            ["powershell", "-Command",
             "Get-WmiObject Win32_DiskDrive | Select-Object DeviceID, Index, "
             "SerialNumber, FirmwareRevision, Model, Manufacturer, Size, MediaType | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                if isinstance(data, dict):
                    data = [data]
                for drive in data:
                    index = drive.get("Index")
                    if index is not None:
                        media = (drive.get("MediaType") or "").lower()
                        drive_type = "HDD"
                        if "ssd" in media or "solid" in media:
                            drive_type = "SSD"
                        elif "removable" in media:
                            drive_type = "Removable"
                        wmi_drives[index] = {
                            "model": drive.get("Model") or "unknown",
                            "manufacturer": drive.get("Manufacturer") or "unknown",
                            "serial": (drive.get("SerialNumber") or "").strip() or None,
                            "firmware": (drive.get("FirmwareRevision") or "").strip() or None,
                            "listed_cap_b": drive.get("Size"),
                            "type": drive_type,
                        }
            except json.JSONDecodeError:
                self.logger.debug("Failed to parse WMI drive data.")

        # Get partition info from PowerShell
        partitions_by_disk: dict[int, list[dict[str, Any]]] = {}
        result = self._run_command(
            ["powershell", "-Command",
             "Get-Partition | Select-Object DiskNumber, PartitionNumber, DriveLetter, "
             "Size, Type, GptType | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                if isinstance(data, dict):
                    data = [data]
                for part in data:
                    disk_num = part.get("DiskNumber")
                    if disk_num is None:
                        continue
                    part_entry: dict[str, Any] = {
                        "name": f"Disk {disk_num} Partition {part.get('PartitionNumber', 0)}",
                    }
                    if part.get("PartitionNumber"):
                        part_entry["number"] = part["PartitionNumber"]
                    if part.get("DriveLetter"):
                        part_entry["label"] = f"{part['DriveLetter']}:"
                    if part.get("Size"):
                        part_entry["size_b"] = part["Size"]
                    if part.get("GptType"):
                        part_entry["type_guid"] = part["GptType"]
                    part_type = (part.get("Type") or "").lower()
                    if "system" in part_type:
                        part_entry["content"] = "filesystem"
                    elif "recovery" in part_type:
                        part_entry["content"] = "filesystem"
                    elif "reserved" in part_type:
                        part_entry["content"] = "unknown"
                    else:
                        part_entry["content"] = "filesystem"
                    part_entry["source"] = "powershell"
                    partitions_by_disk.setdefault(disk_num, []).append(part_entry)
            except json.JSONDecodeError:
                self.logger.debug("Failed to parse partition data.")

        # Get disk health status from Get-PhysicalDisk
        disk_health: dict[int, dict[str, Any]] = {}
        result = self._run_command(
            ["powershell", "-Command",
             "Get-PhysicalDisk | Select-Object DeviceId, FriendlyName, MediaType, "
             "HealthStatus, OperationalStatus, SpindleSpeed | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                if isinstance(data, dict):
                    data = [data]
                for disk in data:
                    dev_id = disk.get("DeviceId")
                    if dev_id is not None:
                        try:
                            disk_num = int(dev_id)
                        except (ValueError, TypeError):
                            continue
                        health_status = disk.get("HealthStatus") or "Unknown"
                        op_status = disk.get("OperationalStatus") or "Unknown"
                        # Determine drive type from MediaType or SpindleSpeed
                        media_type = (disk.get("MediaType") or "").strip()
                        spindle = disk.get("SpindleSpeed")
                        if media_type == "SSD" or spindle == 0:
                            drive_type = "SSD"
                        elif media_type == "HDD" or (spindle and spindle > 0):
                            drive_type = "HDD"
                        else:
                            drive_type = None  # Keep existing type

                        smart: dict[str, Any] = {
                            "overall_health": "PASSED" if health_status == "Healthy" else health_status,
                        }
                        if op_status != "OK":
                            smart["operational_status"] = op_status

                        disk_health[disk_num] = {
                            "smart": smart,
                            "drive_type": drive_type,
                        }
            except json.JSONDecodeError:
                self.logger.debug("Failed to parse disk health data.")

        # Try to get detailed SMART counters (requires admin)
        result = self._run_command(
            ["powershell", "-Command",
             "Get-PhysicalDisk | Get-StorageReliabilityCounter | "
             "Select-Object DeviceId, Temperature, Wear, ReadErrorsTotal, "
             "WriteErrorsTotal, PowerOnHours | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                if isinstance(data, dict):
                    data = [data]
                for counter in data:
                    dev_id = counter.get("DeviceId")
                    if dev_id is not None:
                        try:
                            disk_num = int(dev_id)
                        except (ValueError, TypeError):
                            continue
                        if disk_num in disk_health:
                            smart = disk_health[disk_num]["smart"]
                            if counter.get("Temperature"):
                                smart["temperature_c"] = counter["Temperature"]
                            if counter.get("Wear") is not None:
                                # Wear is percentage used (0-100), convert to remaining
                                smart["wear_leveling_pct"] = 100 - counter["Wear"]
                            if counter.get("ReadErrorsTotal"):
                                smart["read_errors"] = counter["ReadErrorsTotal"]
                            if counter.get("WriteErrorsTotal"):
                                smart["write_errors"] = counter["WriteErrorsTotal"]
                            if counter.get("PowerOnHours"):
                                smart["power_on_hours"] = counter["PowerOnHours"]
            except json.JSONDecodeError:
                self.logger.debug("Failed to parse storage reliability counters.")

        # Build metadata for each drive
        for drive_name in io_stats.keys():
            # Extract disk number from name (e.g., "PhysicalDrive0" -> 0)
            disk_num = None
            if drive_name.startswith("PhysicalDrive"):
                try:
                    disk_num = int(drive_name[13:])
                except ValueError:
                    pass

            if disk_num is not None and disk_num in wmi_drives:
                wmi_info = wmi_drives[disk_num]
                drive_type = wmi_info["type"]
                # Prefer drive type from Get-PhysicalDisk (more reliable)
                if disk_num in disk_health and disk_health[disk_num].get("drive_type"):
                    drive_type = disk_health[disk_num]["drive_type"]
                metadata[drive_name] = {
                    "manufacturer": wmi_info["manufacturer"],
                    "model": wmi_info["model"],
                    "type": drive_type,
                    "serial": wmi_info["serial"],
                    "firmware": wmi_info["firmware"],
                    "listed_cap_b": wmi_info["listed_cap_b"],
                }
                if disk_num in partitions_by_disk:
                    metadata[drive_name]["partitions"] = partitions_by_disk[disk_num]
                # Add SMART data if available
                if disk_num in disk_health:
                    metadata[drive_name]["smart"] = disk_health[disk_num]["smart"]
            else:
                metadata[drive_name] = {
                    "manufacturer": "unknown",
                    "model": drive_name,
                    "type": "unknown",
                }

        self.logger.debug("Created Windows drive metadata for %d drives", len(metadata))
        return metadata

    def _windows_partition_map(self) -> dict[str, int]:
        """Map drive letters to physical disk numbers on Windows."""
        result = self._run_command(
            ["powershell", "-Command",
             "Get-Partition | Select-Object DriveLetter, DiskNumber | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if not result:
            return {}
        try:
            data = json.loads(result)
            # Ensure it's a list (single result comes as dict)
            if isinstance(data, dict):
                data = [data]
            mapping: dict[str, int] = {}
            for entry in data:
                letter = entry.get("DriveLetter")
                disk_num = entry.get("DiskNumber")
                if letter and disk_num is not None:
                    mapping[f"{letter}:\\"] = disk_num
            return mapping
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse Windows partition mapping.")
            return {}

    def _windows_bitlocker_status(self, mountpoints: list[str]) -> dict[str, dict[str, Any]]:
        """Get BitLocker encryption status for Windows volumes.

        Uses Shell COM object to check BitLocker status without requiring admin rights.
        BitLocker protection values:
        - 0/null: Not encrypted or not BitLocker capable
        - 1: BitLocker On (unlocked)
        - 2: BitLocker Off
        - 3: BitLocker suspended
        - 6: BitLocker On (locked)
        """
        if not mountpoints:
            return {}
        # Build PowerShell command to check all mountpoints
        drives_array = ", ".join(f'"{m}"' for m in mountpoints)
        ps_script = (
            f"$shell = New-Object -ComObject Shell.Application; "
            f"@({drives_array}) | ForEach-Object {{ "
            f"$ns = $shell.NameSpace($_); "
            f"if ($ns) {{ $status = $ns.Self.ExtendedProperty('System.Volume.BitLockerProtection'); "
            f"[PSCustomObject]@{{Drive=$_; BitLocker=$status}} }} }} | ConvertTo-Json"
        )
        result = self._run_command(
            ["powershell", "-Command", ps_script],
            stderr=subprocess.DEVNULL,
        )
        if not result:
            return {}
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                data = [data]
            status: dict[str, dict[str, Any]] = {}
            for entry in data:
                mount = entry.get("Drive")
                bitlocker_val = entry.get("BitLocker")
                if not mount:
                    continue
                # Values 1, 3, 6 indicate BitLocker is present
                # 1 = On (unlocked), 3 = Suspended, 6 = On (locked)
                if bitlocker_val in (1, 3, 6):
                    status[mount] = {"encrypted": True, "scheme": "bitlocker"}
            return status
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse BitLocker status.")
            return {}

    def _windows_volume_metadata(self) -> dict[str, dict[str, Any]]:
        """Get volume metadata (label, uuid, fstype) from Windows."""
        result = self._run_command(
            ["powershell", "-Command",
             "Get-Volume | Where-Object { $_.DriveLetter } | "
             "Select-Object DriveLetter, FileSystemLabel, FileSystemType, UniqueId | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if not result:
            return {}
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                data = [data]
            metadata: dict[str, dict[str, Any]] = {}
            for vol in data:
                letter = vol.get("DriveLetter")
                if not letter:
                    continue
                mountpoint = f"{letter}:\\"
                entry: dict[str, Any] = {}
                if vol.get("FileSystemLabel"):
                    entry["label"] = vol["FileSystemLabel"]
                if vol.get("FileSystemType"):
                    entry["fstype"] = vol["FileSystemType"]
                if vol.get("UniqueId"):
                    # Extract UUID from Windows format: \\?\Volume{uuid}\
                    uid = vol["UniqueId"]
                    if "{" in uid and "}" in uid:
                        start = uid.index("{") + 1
                        end = uid.index("}")
                        entry["uuid"] = uid[start:end]
                    else:
                        entry["uuid"] = uid
                if entry:
                    metadata[mountpoint] = entry
            return metadata
        except json.JSONDecodeError:
            self.logger.debug("Failed to parse Windows volume metadata.")
            return {}

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

        # On Linux, use lsblk hierarchy to find all filesystems under each drive
        if platform.system().lower() == "linux":
            self._populate_usage_linux(metadata)
        # On Windows, sum all partition usage for all drives
        elif platform.system().lower() == "windows":
            self._populate_usage_windows(metadata)

    def _populate_usage_linux(self, metadata: dict[str, dict[str, Any]]) -> None:
        """Populate drive usage by aggregating all filesystems under each drive."""
        output = self._run_command(
            [self.config.lsblk_path, "-J", "-b", "-o", "NAME,TYPE,MOUNTPOINT,SIZE"]
        )
        if output is None:
            return
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return

        def collect_mountpoints(device: dict[str, Any]) -> list[str]:
            """Recursively collect all mountpoints under a device."""
            mountpoints: list[str] = []
            mp = device.get("mountpoint")
            if mp:
                mountpoints.append(mp)
            for child in device.get("children", []):
                mountpoints.extend(collect_mountpoints(child))
            return mountpoints

        for block in data.get("blockdevices", []):
            if block.get("type") != "disk":
                continue
            name = block.get("name")
            if not name or name not in metadata:
                continue

            # Find all mountpoints under this drive
            mountpoints = collect_mountpoints(block)
            total_used = 0
            total_free = 0

            for mp in mountpoints:
                try:
                    usage = psutil.disk_usage(mp)
                    total_used += int(usage.used)
                    total_free += int(usage.free)
                except OSError:
                    continue

            metadata[name]["used_b"] = total_used
            metadata[name]["available_b"] = total_free

    def _populate_usage_windows(self, metadata: dict[str, dict[str, Any]]) -> None:
        """Populate drive usage on Windows by summing all partitions."""
        partitions = psutil.disk_partitions(all=False)
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
            # For NVMe, use controller device (nvme0) instead of namespace (nvme0n1)
            # as smartctl works better with the controller for SMART data
            if name.startswith("nvme") and "n" in name:
                ctrl_name = re.sub(r"^(nvme\d+)n\d+$", r"\1", name)
                device_path = f"/dev/{ctrl_name}"
            else:
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
            # Convert critical_warning to hex string as per schema
            crit_warn = nvme.get("critical_warning")
            crit_warn_str = f"0x{crit_warn:02x}" if crit_warn is not None else None
            mapping = {
                "power_on_hours": nvme.get("power_on_hours"),
                "critical_warning": crit_warn_str,
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
        else:
            # ATA/SATA drive - extract from top-level fields and attributes
            if "power_on_time" in data:
                hours = data["power_on_time"].get("hours")
                if hours is not None:
                    smart["power_on_hours"] = hours
            if "power_cycle_count" in data:
                smart["power_cycles"] = data["power_cycle_count"]
            if "temperature" in data:
                temp = data["temperature"].get("current")
                if temp is not None:
                    smart["temperature_c"] = temp
            # Extract key SMART attributes
            ata_attrs = data.get("ata_smart_attributes", {}).get("table", [])
            attr_map = {
                5: "reallocated_sector_ct",
                197: "current_pending_sector",
                198: "offline_uncorrectable",
            }
            for attr in ata_attrs:
                attr_id = attr.get("id")
                if attr_id in attr_map:
                    raw_val = attr.get("raw", {}).get("value")
                    if raw_val is not None:
                        smart[attr_map[attr_id]] = raw_val
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

        freebsd_sensors = self._collect_freebsd_sensors()
        if freebsd_sensors:
            self._merge_motherboard(motherboard, freebsd_sensors)

        # Try sysfs DMI first (no root required), then fall back to dmidecode
        sysfs_dmi = self._collect_sysfs_dmi()
        if sysfs_dmi:
            motherboard.update(sysfs_dmi)

        dmi_board = self._collect_dmidecode()
        if dmi_board:
            motherboard.update(dmi_board)

        wmi_board = self._collect_wmi_motherboard()
        if wmi_board:
            motherboard.update(wmi_board)

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
                # First pass: collect all values for this sensor
                sensor_values: dict[str, float] = {}
                for key, value in readings.items():
                    if isinstance(value, (int, float)):
                        sensor_values[key] = float(value)

                entry_name = f"{chip} {label}"

                # Process temperature sensors with max values
                temp_input = None
                temp_max = None
                for key, value in sensor_values.items():
                    if key.endswith("_input") and key.startswith("temp"):
                        temp_input = value
                    elif key.endswith("_max") and key.startswith("temp"):
                        temp_max = value

                if temp_input is not None:
                    temp_entry: dict[str, Any] = {
                        "name": entry_name,
                        "temp_c": temp_input,
                        "source": "sensors",
                    }
                    if temp_max is not None:
                        temp_entry["max_c"] = temp_max
                    temps.append(temp_entry)

                # Process other sensors
                for key, value in sensor_values.items():
                    if not key.endswith("_input"):
                        continue
                    if key.startswith("temp"):
                        continue  # Already processed above
                    if key.startswith("fan"):
                        fans.append(
                            {
                                "name": entry_name,
                                "rpm": value,
                                "source": "sensors",
                            }
                        )
                    elif key.startswith("in"):
                        voltages.append(
                            {
                                "name": entry_name,
                                "voltage_v": value,
                                "source": "sensors",
                            }
                        )
                    elif key.startswith("curr"):
                        currents.append(
                            {
                                "name": entry_name,
                                "current_a": value,
                                "source": "sensors",
                            }
                        )
                    elif key.startswith("power"):
                        powers.append(
                            {
                                "name": entry_name,
                                "power_w": value,
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

    def _get_drive_temp_from_sensors(self, drive_name: str) -> float | None:
        """Get drive temperature from Linux sensors for NVMe/SATA drives."""
        output = self._run_command([self.config.sensors_path, "-j"])
        if output is None:
            return None
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return None

        # Look for nvme or drivetemp sensors matching the drive
        for chip, chip_data in data.items():
            if not isinstance(chip_data, dict):
                continue
            # Match nvme drives: nvme-pci-XXXX for nvmeXnY drives
            if chip.startswith("nvme-") and drive_name.startswith("nvme"):
                # Look for Composite temp (main drive temp)
                for label, readings in chip_data.items():
                    if not isinstance(readings, dict):
                        continue
                    if label.lower() == "composite":
                        for key, value in readings.items():
                            if key.endswith("_input") and isinstance(value, (int, float)):
                                return float(value)
            # Match SATA drives via drivetemp
            elif chip.startswith("drivetemp-") and not drive_name.startswith("nvme"):
                for label, readings in chip_data.items():
                    if not isinstance(readings, dict):
                        continue
                    for key, value in readings.items():
                        if key.endswith("_input") and isinstance(value, (int, float)):
                            return float(value)
        return None

    def _collect_sysfs_dmi(self) -> dict[str, Any] | None:
        """Collect motherboard info from /sys/class/dmi/id/ (no root required)."""
        if platform.system().lower() != "linux":
            return None

        dmi_path = Path("/sys/class/dmi/id")
        if not dmi_path.exists():
            return None

        result: dict[str, Any] = {}

        # Read board info
        board_vendor = self._read_sysfs_file(dmi_path / "board_vendor")
        board_name = self._read_sysfs_file(dmi_path / "board_name")
        if board_vendor:
            result["manufacturer"] = board_vendor
        if board_name:
            result["name"] = board_name

        # Read BIOS info
        bios_version = self._read_sysfs_file(dmi_path / "bios_version")
        bios_date = self._read_sysfs_file(dmi_path / "bios_date")
        if bios_version:
            result["bios_version"] = bios_version
        if bios_date:
            result["bios_date"] = bios_date

        # Read chassis type
        chassis_type_str = self._read_sysfs_file(dmi_path / "chassis_type")
        if chassis_type_str:
            try:
                chassis_type = int(chassis_type_str)
                chassis_names = {
                    1: "Other", 2: "Unknown", 3: "Desktop", 4: "Low Profile Desktop",
                    5: "Pizza Box", 6: "Mini Tower", 7: "Tower", 8: "Portable",
                    9: "Laptop", 10: "Notebook", 11: "Hand Held", 12: "Docking Station",
                    13: "All in One", 14: "Sub Notebook", 15: "Space-saving",
                    16: "Lunch Box", 17: "Main Server Chassis", 18: "Expansion Chassis",
                    19: "SubChassis", 20: "Bus Expansion Chassis", 21: "Peripheral Chassis",
                    22: "RAID Chassis", 23: "Rack Mount Chassis", 24: "Sealed-case PC",
                    25: "Multi-system chassis", 26: "Compact PCI", 27: "Advanced TCA",
                    28: "Blade", 29: "Blade Enclosure", 30: "Tablet", 31: "Convertible",
                    32: "Detachable", 33: "IoT Gateway", 34: "Embedded PC", 35: "Mini PC",
                    36: "Stick PC",
                }
                result["chassis_type"] = chassis_names.get(chassis_type, f"Type {chassis_type}")
            except ValueError:
                pass

        return result if result else None

    def _read_sysfs_file(self, path: Path) -> str | None:
        """Read a sysfs file and return its contents stripped, or None if unreadable."""
        try:
            return path.read_text().strip() or None
        except (OSError, PermissionError):
            return None

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

    def _collect_wmi_motherboard(self) -> dict[str, Any] | None:
        """Get motherboard and BIOS info from WMI on Windows."""
        if platform.system().lower() != "windows":
            return None

        motherboard: dict[str, Any] = {}

        # Get baseboard info
        result = self._run_command(
            ["powershell", "-Command",
             "Get-WmiObject Win32_BaseBoard | Select-Object Manufacturer, Product | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                if data.get("Manufacturer"):
                    motherboard["manufacturer"] = data["Manufacturer"]
                if data.get("Product"):
                    motherboard["name"] = data["Product"]
            except json.JSONDecodeError:
                pass

        # Get BIOS info
        result = self._run_command(
            ["powershell", "-Command",
             "Get-WmiObject Win32_BIOS | Select-Object SMBIOSBIOSVersion, ReleaseDate | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                if data.get("SMBIOSBIOSVersion"):
                    motherboard["bios_version"] = data["SMBIOSBIOSVersion"]
                if data.get("ReleaseDate"):
                    # Parse WMI date format: 20250917000000.000000+000
                    raw_date = data["ReleaseDate"]
                    if raw_date and len(raw_date) >= 8:
                        try:
                            motherboard["bios_date"] = f"{raw_date[4:6]}/{raw_date[6:8]}/{raw_date[0:4]}"
                        except (IndexError, ValueError):
                            pass
            except json.JSONDecodeError:
                pass

        return motherboard if motherboard else None

    def _collect_host_hardware_windows(self) -> dict[str, Any] | None:
        """Get host hardware info (model, serial, chassis type) from WMI on Windows."""
        if platform.system().lower() != "windows":
            return None

        result_info: dict[str, Any] = {}

        # Get computer model from Win32_ComputerSystem
        result = self._run_command(
            ["powershell", "-Command",
             "Get-WmiObject Win32_ComputerSystem | Select-Object Model, Manufacturer | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                model = data.get("Model")
                manufacturer = data.get("Manufacturer")
                if model and manufacturer:
                    result_info["model"] = f"{manufacturer} {model}"
                elif model:
                    result_info["model"] = model
            except json.JSONDecodeError:
                pass

        # Get serial number and chassis type from Win32_SystemEnclosure
        result = self._run_command(
            ["powershell", "-Command",
             "Get-WmiObject Win32_SystemEnclosure | Select-Object SerialNumber, ChassisTypes | ConvertTo-Json"],
            stderr=subprocess.DEVNULL,
        )
        if result:
            try:
                data = json.loads(result)
                serial = data.get("SerialNumber")
                if serial and serial.strip():
                    result_info["serial"] = serial.strip()

                chassis_types = data.get("ChassisTypes")
                if chassis_types and isinstance(chassis_types, list) and len(chassis_types) > 0:
                    # SMBIOS chassis type mapping
                    chassis_names = {
                        1: "Other", 2: "Unknown", 3: "Desktop", 4: "Low Profile Desktop",
                        5: "Pizza Box", 6: "Mini Tower", 7: "Tower", 8: "Portable",
                        9: "Laptop", 10: "Notebook", 11: "Hand Held", 12: "Docking Station",
                        13: "All in One", 14: "Sub Notebook", 15: "Space-saving",
                        16: "Lunch Box", 17: "Main Server Chassis", 18: "Expansion Chassis",
                        19: "SubChassis", 20: "Bus Expansion Chassis", 21: "Peripheral Chassis",
                        22: "RAID Chassis", 23: "Rack Mount Chassis", 24: "Sealed-case PC",
                        25: "Multi-system chassis", 26: "Compact PCI", 27: "Advanced TCA",
                        28: "Blade", 29: "Blade Enclosure", 30: "Tablet", 31: "Convertible",
                        32: "Detachable", 33: "IoT Gateway", 34: "Embedded PC", 35: "Mini PC",
                        36: "Stick PC",
                    }
                    chassis_type = chassis_types[0]
                    result_info["chassis_type"] = chassis_names.get(chassis_type, f"Type {chassis_type}")
            except json.JSONDecodeError:
                pass

        return result_info if result_info else None

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

            # Add memory info
            memory_entry: dict[str, Any] = {}
            if device.get("mem_clock_hz") is not None:
                memory_entry["clock_hz"] = device["mem_clock_hz"]

            # Build memory types array from different memory regions
            mem_types: list[dict[str, Any]] = []

            # Dedicated/main GPU memory
            if device.get("memory"):
                mem = device["memory"]
                mem_type: dict[str, Any] = {"name": "GPU Memory"}
                if "used_b" in mem:
                    mem_type["used_b"] = mem["used_b"]
                if "available_b" in mem:
                    mem_type["available_b"] = mem["available_b"]
                elif "total_b" in mem and "used_b" in mem:
                    mem_type["available_b"] = mem["total_b"] - mem["used_b"]
                if "total_b" in mem:
                    mem_type["total_b"] = mem["total_b"]
                if "used_b" in mem_type and "available_b" in mem_type:
                    mem_types.append(mem_type)

            # Shared system memory
            if device.get("memory_shared"):
                mem = device["memory_shared"]
                mem_type = {"name": "Shared Memory"}
                if "used_b" in mem:
                    mem_type["used_b"] = mem["used_b"]
                    mem_type["available_b"] = 0  # Shared memory is dynamically allocated
                    mem_types.append(mem_type)

            # D3D dedicated memory
            if device.get("memory_d3d"):
                mem = device["memory_d3d"]
                mem_type = {"name": "D3D Dedicated"}
                if "used_b" in mem:
                    mem_type["used_b"] = mem["used_b"]
                    mem_type["available_b"] = 0
                    mem_types.append(mem_type)

            if mem_types:
                memory_entry["types"] = mem_types
            if memory_entry:
                entry["memory"] = memory_entry

            gpus.append(entry)
        return gpus

    def _collect_intel_gpu(self) -> dict[str, Any] | None:
        """Collect Intel iGPU metrics using intel_gpu_top."""
        if platform.system().lower() != "linux":
            return None
        # intel_gpu_top runs continuously, so we need to use timeout
        try:
            result = subprocess.run(
                [self.config.intel_gpu_top_path, "-J", "-s", "500", "-o", "-"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            output = result.stdout
        except subprocess.TimeoutExpired as e:
            # This is expected - intel_gpu_top runs until killed
            output = e.stdout.decode("utf-8") if e.stdout else None
        except FileNotFoundError:
            self.logger.debug("intel_gpu_top not found.")
            return None
        except Exception as e:
            self.logger.debug("intel_gpu_top failed: %s", e)
            return None
        if not output:
            self.logger.debug("intel_gpu_top produced no output.")
            return None
        # intel_gpu_top outputs multiple JSON objects, take the last complete one
        lines = output.strip().split("\n")
        last_obj = None
        brace_count = 0
        obj_start = -1
        for i, line in enumerate(lines):
            brace_count += line.count("{") - line.count("}")
            if "{" in line and obj_start < 0:
                obj_start = i
            if brace_count == 0 and obj_start >= 0:
                try:
                    obj_str = "\n".join(lines[obj_start : i + 1])
                    last_obj = json.loads(obj_str)
                except json.JSONDecodeError:
                    pass
                obj_start = -1
        if not last_obj:
            self.logger.debug("Failed to parse intel_gpu_top JSON output.")
            return None

        # Build GPU entry
        entry: dict[str, Any] = {"name": "Intel Integrated Graphics"}

        # RC6 is idle percentage, so utilization = 100 - rc6
        rc6 = last_obj.get("rc6", {}).get("value", 0)
        core_load = max(0, min(100, 100 - rc6))

        core: dict[str, Any] = {"name": "Core", "load_pct": round(core_load, 1)}

        # Frequency (convert MHz to Hz)
        freq = last_obj.get("frequency", {})
        actual_mhz = freq.get("actual", 0)
        if actual_mhz:
            core["clock_hz"] = int(actual_mhz * 1_000_000)

        # Power
        power = last_obj.get("power", {})
        gpu_power = power.get("GPU")
        if gpu_power is not None:
            core["power_w"] = round(gpu_power, 2)

        entry["core"] = core

        # Engines
        engines_data = last_obj.get("engines", {})
        engines: list[dict[str, Any]] = []
        for engine_name, engine_info in engines_data.items():
            busy = engine_info.get("busy", 0)
            engines.append({"name": engine_name, "load_pct": round(busy, 1)})
        if engines:
            entry["engines"] = engines
        else:
            entry["engines"] = [{"name": "Core", "load_pct": core["load_pct"]}]

        return entry

    def _collect_tpus(self) -> list[dict[str, Any]]:
        """Collect TPU metrics from Google Coral Edge TPU via sysfs."""
        if platform.system().lower() != "linux":
            return []
        if not self.config.enable_tpu:
            return []

        tpus: list[dict[str, Any]] = []
        apex_path = Path("/sys/class/apex")
        if not apex_path.exists():
            self.logger.debug("No TPU devices found (apex sysfs not present).")
            return []

        for apex_dir in sorted(apex_path.iterdir()):
            if not apex_dir.is_dir():
                continue
            name = apex_dir.name
            entry: dict[str, Any] = {"name": name}

            # Read device type
            device_type_file = apex_dir / "device_type"
            if device_type_file.exists():
                try:
                    device_type = device_type_file.read_text().strip()
                    if device_type == "apex":
                        entry["vendor"] = "Google"
                        entry["model"] = "Coral Edge TPU"
                except OSError:
                    pass

            # Read temperature (in millicelsius)
            temp_file = apex_dir / "temp"
            if temp_file.exists():
                try:
                    temp_mc = int(temp_file.read_text().strip())
                    entry["temp_c"] = round(temp_mc / 1000, 1)
                except (OSError, ValueError):
                    pass

            # Build thermal info with trip points
            thermal: dict[str, Any] = {}
            if "temp_c" in entry:
                thermal["temp_c"] = entry["temp_c"]

            # Trip points (thermal throttling thresholds)
            trip_points = []
            for i in range(3):
                trip_file = apex_dir / f"trip_point{i}_temp"
                if trip_file.exists():
                    try:
                        trip_mc = int(trip_file.read_text().strip())
                        trip_points.append(trip_mc / 1000)
                    except (OSError, ValueError):
                        pass

            if trip_points:
                # First trip point is typically warning, last is critical
                thermal["warning_c"] = trip_points[0]
                if len(trip_points) > 1:
                    thermal["critical_c"] = trip_points[-1]

            # Hardware warning thresholds
            hw_warn1_file = apex_dir / "hw_temp_warn1"
            hw_warn1_en_file = apex_dir / "hw_temp_warn1_en"
            if hw_warn1_file.exists() and hw_warn1_en_file.exists():
                try:
                    enabled = int(hw_warn1_en_file.read_text().strip())
                    if enabled:
                        warn_mc = int(hw_warn1_file.read_text().strip())
                        thermal.setdefault("critical_c", warn_mc / 1000)
                except (OSError, ValueError):
                    pass

            # Check if throttling (temp above first trip point)
            if "temp_c" in entry and "warning_c" in thermal:
                thermal["throttling"] = entry["temp_c"] >= thermal["warning_c"]

            if thermal:
                entry["thermal"] = thermal

            # Read status
            status_file = apex_dir / "status"
            if status_file.exists():
                try:
                    status = status_file.read_text().strip()
                    # If not ALIVE, report as issue
                    if status != "ALIVE":
                        self.logger.warning("TPU %s status: %s", name, status)
                except OSError:
                    pass

            tpus.append(entry)

        return tpus

    def _collect_backup_providers(self) -> list[dict[str, Any]]:
        """Collect Borg backup repository status."""
        if not self.config.borg_repos:
            return []

        providers: list[dict[str, Any]] = []
        for repo in self.config.borg_repos:
            entry: dict[str, Any] = {
                "name": Path(repo).name or repo,
                "type": "borg",
                "repo": repo,
            }

            # Get repository info
            info_output = self._run_command(
                [self.config.borg_path, "info", "--json", repo],
                stderr=subprocess.DEVNULL,
            )
            if info_output:
                try:
                    info = json.loads(info_output)
                    repo_info = info.get("repository", {})
                    if "last_modified" in repo_info:
                        entry["last_success_ts"] = repo_info["last_modified"]
                    cache = info.get("cache", {})
                    stats = cache.get("stats", {})
                    if "total_size" in stats:
                        entry["repo_size_b"] = stats["total_size"]
                    if "total_csize" in stats:
                        entry["compression"] = f"ratio {stats.get('total_size', 0) / max(stats.get('total_csize', 1), 1):.2f}"
                    # Check encryption
                    enc = info.get("encryption", {})
                    if enc.get("mode"):
                        entry["encryption"] = enc["mode"]
                except json.JSONDecodeError:
                    self.logger.debug("Failed to parse borg info JSON for %s", repo)

            # Get list of archives to find the latest
            list_output = self._run_command(
                [self.config.borg_path, "list", "--json", repo],
                stderr=subprocess.DEVNULL,
            )
            if list_output:
                try:
                    list_data = json.loads(list_output)
                    archives = list_data.get("archives", [])
                    if archives:
                        entry["retention"] = {"snapshots": len(archives)}
                        # Find most recent archive
                        latest = max(archives, key=lambda a: a.get("time", ""))
                        if latest.get("time"):
                            entry["last_success_ts"] = latest["time"]
                            # Calculate age
                            try:
                                last_dt = datetime.fromisoformat(
                                    latest["time"].replace("Z", "+00:00")
                                )
                                now = datetime.now(timezone.utc)
                                entry["age_s"] = int((now - last_dt).total_seconds())
                            except ValueError:
                                pass
                except json.JSONDecodeError:
                    self.logger.debug("Failed to parse borg list JSON for %s", repo)

            # Determine health: ok if we could read the repo and it's not too old
            entry["ok"] = info_output is not None
            if entry.get("age_s", 0) > 86400 * 7:  # Warn if older than 7 days
                entry["ok"] = False
                entry["last_status"] = "stale"
            elif entry["ok"]:
                entry["last_status"] = "success"
            else:
                entry["last_status"] = "unknown"

            providers.append(entry)

        return providers

    def _collect_time_server(self) -> dict[str, Any] | None:
        """Collect time server metrics from chrony, gpsd, and PPS."""
        result: dict[str, Any] = {}

        # Collect chrony tracking info
        tracking = self._collect_chrony_tracking()
        if tracking:
            result["tracking"] = tracking

        # Collect chrony sources
        sources = self._collect_chrony_sources()
        if sources:
            result["sources"] = sources

        # Collect server stats (may require elevated privileges)
        server_stats = self._collect_chrony_serverstats()
        if server_stats:
            result["server_stats"] = server_stats

        # Collect GPS data from gpsd
        gps = self._collect_gpsd()
        if gps:
            result["gps"] = gps

        # Collect PPS data
        pps = self._collect_pps()
        if pps:
            result["pps"] = pps

        # Collect service statuses
        services = self._collect_time_services()
        if services:
            result["services"] = services

        return result if result else None

    def _collect_chrony_tracking(self) -> dict[str, Any] | None:
        """Collect chrony tracking status."""
        output = self._run_command([self.config.chronyc_path, "tracking"])
        if not output:
            return None

        result: dict[str, Any] = {}
        for line in output.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "Reference ID":
                # Parse "E1FE1EBE (time.cloudflare.com)"
                parts = value.split("(")
                result["reference_id"] = parts[0].strip()
                if len(parts) > 1:
                    result["reference_name"] = parts[1].rstrip(")")
            elif key == "Stratum":
                try:
                    result["stratum"] = int(value)
                except ValueError:
                    pass
            elif key == "System time":
                # Parse "0.000226142 seconds slow of NTP time"
                try:
                    parts = value.split()
                    offset = float(parts[0])
                    if "slow" in value:
                        offset = -offset
                    result["system_time_offset_s"] = offset
                except (ValueError, IndexError):
                    pass
            elif key == "Last offset":
                try:
                    result["last_offset_s"] = float(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == "RMS offset":
                try:
                    result["rms_offset_s"] = float(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == "Frequency":
                # Parse "14.770 ppm fast"
                try:
                    parts = value.split()
                    freq = float(parts[0])
                    if "fast" in value:
                        pass  # positive
                    elif "slow" in value:
                        freq = -freq
                    result["frequency_ppm"] = freq
                except (ValueError, IndexError):
                    pass
            elif key == "Residual freq":
                try:
                    result["residual_freq_ppm"] = float(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == "Skew":
                try:
                    result["skew_ppm"] = float(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == "Root delay":
                try:
                    result["root_delay_s"] = float(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == "Root dispersion":
                try:
                    result["root_dispersion_s"] = float(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == "Update interval":
                try:
                    result["update_interval_s"] = float(value.split()[0])
                except (ValueError, IndexError):
                    pass
            elif key == "Leap status":
                result["leap_status"] = value

        return result if result else None

    def _collect_chrony_sources(self) -> list[dict[str, Any]] | None:
        """Collect chrony sources and sourcestats."""
        # Get sources
        sources_output = self._run_command([self.config.chronyc_path, "sources"])
        if not sources_output:
            return None

        # Get sourcestats for additional metrics
        stats_output = self._run_command([self.config.chronyc_path, "sourcestats"])
        stats_map: dict[str, dict[str, Any]] = {}
        if stats_output:
            for line in stats_output.strip().splitlines():
                if line.startswith("=") or line.startswith(" "):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    name = parts[0]
                    try:
                        stats_map[name] = {
                            "samples": int(parts[1]),
                            "frequency_ppm": float(parts[4]),
                            "frequency_skew_ppm": float(parts[5]),
                            "std_dev_s": self._parse_time_value(parts[7]),
                        }
                    except (ValueError, IndexError):
                        pass

        sources: list[dict[str, Any]] = []
        mode_map = {"^": "server", "=": "peer", "#": "local"}
        state_map = {
            "*": "sync",
            "+": "combined",
            "-": "not_combined",
            "x": "maybe_error",
            "~": "too_variable",
            "?": "unusable",
            " ": "unselected",
        }

        for line in sources_output.strip().splitlines():
            # Skip header lines
            if line.startswith("=") or line.startswith(" ") or "Name/IP" in line:
                continue
            if len(line) < 3:
                continue

            # Parse "MS Name/IP address         Stratum Poll Reach LastRx Last sample"
            mode_char = line[0]
            state_char = line[1]

            # Parse the rest
            rest = line[2:].strip()
            parts = rest.split()
            if len(parts) < 5:
                continue

            name = parts[0]
            entry: dict[str, Any] = {
                "name": name,
                "mode": mode_map.get(mode_char, "server"),
                "state": state_map.get(state_char, "unselected"),
            }

            try:
                entry["stratum"] = int(parts[1])
            except (ValueError, IndexError):
                pass

            try:
                entry["poll_interval"] = int(parts[2])
            except (ValueError, IndexError):
                pass

            try:
                entry["reachability"] = int(parts[3], 8)  # Octal
            except (ValueError, IndexError):
                pass

            # Parse LastRx (can be a number or "6d" for days, "21m" for minutes, etc.)
            if len(parts) > 4:
                entry["last_rx_s"] = self._parse_time_interval(parts[4])

            # Parse last sample offset and error from the "[xxxx] +/- zzzz" format
            sample_match = line.find("[")
            if sample_match != -1:
                sample_end = line.find("]", sample_match)
                if sample_end != -1:
                    sample_str = line[sample_match + 1:sample_end].strip()
                    entry["offset_s"] = self._parse_time_value(sample_str)

                    # Find error after "+/-"
                    error_start = line.find("+/-", sample_end)
                    if error_start != -1:
                        error_str = line[error_start + 3:].strip()
                        entry["error_s"] = self._parse_time_value(error_str)

            # Merge stats if available
            if name in stats_map:
                entry.update(stats_map[name])

            sources.append(entry)

        return sources if sources else None

    def _collect_chrony_serverstats(self) -> dict[str, Any] | None:
        """Collect chrony server statistics."""
        output = self._run_command([self.config.chronyc_path, "serverstats"])
        if not output:
            return None

        result: dict[str, Any] = {}
        for line in output.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            try:
                if "NTP packets received" in key:
                    result["ntp_packets_received"] = int(value)
                elif "NTP packets dropped" in key:
                    result["ntp_packets_dropped"] = int(value)
                elif "Command packets received" in key:
                    result["cmd_packets_received"] = int(value)
                elif "Command packets dropped" in key:
                    result["cmd_packets_dropped"] = int(value)
                elif "NTS-KE connections accepted" in key:
                    result["nts_ke_connections"] = int(value)
                elif "Authenticated NTP packets" in key:
                    result["authenticated_packets"] = int(value)
                elif "Interleaved NTP packets" in key:
                    result["interleaved_packets"] = int(value)
            except ValueError:
                pass

        # Try to get client count
        # Use -n to avoid DNS lookups which can hang with many clients
        clients_output = self._run_command(
            [self.config.chronyc_path, "-n", "clients"]
        )
        if clients_output:
            # Count non-header lines
            client_count = 0
            for line in clients_output.strip().splitlines():
                if line.startswith("=") or line.startswith("Hostname") or not line.strip():
                    continue
                client_count += 1
            if client_count > 0:
                result["client_count"] = client_count

        return result if result else None

    def _collect_gpsd(self) -> dict[str, Any] | None:
        """Collect GPS data from gpsd via gpspipe."""
        output = self._run_command(
            [self.config.gpspipe_path, "-w", "-n", "5"],
            stderr=subprocess.DEVNULL,
        )
        if not output:
            return None

        result: dict[str, Any] = {}
        satellites: list[dict[str, Any]] = []

        for line in output.strip().splitlines():
            try:
                data = json.loads(line)
                msg_class = data.get("class")

                if msg_class == "DEVICES":
                    # Handle DEVICES wrapper with array of devices
                    devices = data.get("devices", [])
                    if devices:
                        dev = devices[0]  # Use first device
                        result["device"] = dev.get("path")
                        result["driver"] = dev.get("driver")
                        subtype = dev.get("subtype1") or dev.get("subtype")
                        if subtype:
                            result["model"] = subtype

                elif msg_class == "DEVICE":
                    # Handle standalone DEVICE message
                    result["device"] = data.get("path")
                    result["driver"] = data.get("driver")
                    subtype = data.get("subtype1") or data.get("subtype")
                    if subtype:
                        result["model"] = subtype

                elif msg_class == "TPV":
                    result["mode"] = data.get("mode", 0)
                    result["status"] = data.get("status", 0)
                    if "time" in data:
                        result["time"] = data["time"]
                    if "lat" in data:
                        result["latitude"] = data["lat"]
                    if "lon" in data:
                        result["longitude"] = data["lon"]
                    if "alt" in data:
                        result["altitude_m"] = data["alt"]
                    if "speed" in data:
                        result["speed_mps"] = data["speed"]
                    if "climb" in data:
                        result["climb_mps"] = data["climb"]
                    if "track" in data:
                        result["track_deg"] = data["track"]
                    if "ept" in data:
                        result["ept_s"] = data["ept"]
                    if "epx" in data:
                        result["epx_m"] = data["epx"]
                    if "epy" in data:
                        result["epy_m"] = data["epy"]
                    if "epv" in data:
                        result["epv_m"] = data["epv"]
                    if "leapseconds" in data:
                        result["leapseconds"] = data["leapseconds"]

                elif msg_class == "SKY":
                    if "hdop" in data and data["hdop"] < 99:
                        result["hdop"] = data["hdop"]
                    if "vdop" in data and data["vdop"] < 99:
                        result["vdop"] = data["vdop"]
                    if "pdop" in data and data["pdop"] < 99:
                        result["pdop"] = data["pdop"]
                    if "tdop" in data and data["tdop"] < 99:
                        result["tdop"] = data["tdop"]
                    if "gdop" in data and data["gdop"] < 99:
                        result["gdop"] = data["gdop"]
                    if "nSat" in data:
                        result["satellites_visible"] = data["nSat"]
                    if "uSat" in data:
                        result["satellites_used"] = data["uSat"]

                    # Collect satellite info
                    for sat in data.get("satellites", []):
                        sat_entry: dict[str, Any] = {}
                        if "PRN" in sat:
                            sat_entry["prn"] = sat["PRN"]
                        if "gnssid" in sat:
                            sat_entry["gnssid"] = sat["gnssid"]
                        if "el" in sat:
                            sat_entry["elevation"] = sat["el"]
                        if "az" in sat:
                            sat_entry["azimuth"] = sat["az"]
                        if "ss" in sat:
                            sat_entry["signal_strength"] = sat["ss"]
                        if "used" in sat:
                            sat_entry["used"] = sat["used"]
                        if "health" in sat:
                            sat_entry["health"] = sat["health"]
                        if sat_entry:
                            satellites.append(sat_entry)

            except json.JSONDecodeError:
                continue

        if satellites:
            result["satellites"] = satellites

        return result if result else None

    def _collect_pps(self) -> dict[str, Any] | None:
        """Collect PPS signal data from sysfs."""
        pps_device = self.config.pps_device or "/dev/pps0"
        device_name = Path(pps_device).name  # e.g., "pps0"
        sysfs_path = f"/sys/class/pps/{device_name}"

        result: dict[str, Any] = {"device": pps_device}

        # Read assert timestamp and sequence
        assert_data = self._read_file(f"{sysfs_path}/assert")
        if assert_data:
            try:
                # Format: "1768068085.000001313#1291306"
                parts = assert_data.strip().split("#")
                if len(parts) == 2:
                    result["assert_timestamp_s"] = float(parts[0])
                    result["assert_sequence"] = int(parts[1])
            except (ValueError, IndexError):
                pass

        # Read clear timestamp and sequence
        clear_data = self._read_file(f"{sysfs_path}/clear")
        if clear_data:
            try:
                parts = clear_data.strip().split("#")
                if len(parts) == 2:
                    result["clear_timestamp_s"] = float(parts[0])
                    result["clear_sequence"] = int(parts[1])
            except (ValueError, IndexError):
                pass

        # Only return if we got some data beyond just the device
        return result if len(result) > 1 else None

    def _collect_time_services(self) -> dict[str, Any] | None:
        """Check status of time-related services."""
        result: dict[str, Any] = {}

        # Check chronyd
        chrony_status = self._run_command(
            [self.config.systemctl_path, "is-active", "chronyd"]
        )
        if chrony_status:
            result["chronyd"] = chrony_status.strip() == "active"

        # Check gpsd
        gpsd_status = self._run_command(
            [self.config.systemctl_path, "is-active", "gpsd"]
        )
        if gpsd_status:
            result["gpsd"] = gpsd_status.strip() == "active"

        # Check str2str (NTRIP client)
        ntrip_status = self._run_command(
            [self.config.systemctl_path, "is-active", "str2str"]
        )
        if ntrip_status:
            result["ntrip"] = ntrip_status.strip() == "active"

        return result if result else None

    def _parse_time_value(self, value: str) -> float:
        """Parse a time value like '21ms', '+1773ns', '-2222us', '15ms' to seconds."""
        value = value.strip().lstrip("+")
        multipliers = {
            "ns": 1e-9,
            "us": 1e-6,
            "ms": 1e-3,
            "s": 1,
        }
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                try:
                    return float(value[:-len(suffix)]) * mult
                except ValueError:
                    return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0

    def _parse_time_interval(self, value: str) -> float:
        """Parse a time interval like '6d', '21m', '33m', '260' to seconds."""
        value = value.strip()
        if value == "-":
            return 0.0

        multipliers = {
            "d": 86400,
            "h": 3600,
            "m": 60,
            "s": 1,
        }
        for suffix, mult in multipliers.items():
            if value.endswith(suffix):
                try:
                    return float(value[:-1]) * mult
                except ValueError:
                    return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0

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
            if not result.stdout:
                return None
        if result.stdout:
            self.logger.log(TRACE_LEVEL, "stdout: %s", result.stdout.strip())
        return result.stdout

    def _read_file(self, path: str) -> str | None:
        """Read a file and return its contents, or None if it doesn't exist."""
        try:
            return Path(path).read_text()
        except (FileNotFoundError, PermissionError, OSError):
            return None

    def _collect_vcgencmd_throttling(self) -> dict[str, Any] | None:
        """Collect Raspberry Pi throttling status from vcgencmd get_throttled."""
        output = self._run_command(["vcgencmd", "get_throttled"])
        if not output:
            return None

        # Parse "throttled=0x50005" format
        try:
            hex_str = output.strip().split("=")[1]
            flags = int(hex_str, 16)
        except (IndexError, ValueError):
            self.logger.debug("Failed to parse vcgencmd get_throttled output: %s", output)
            return None

        # Decode the bitmask
        # Bit meanings:
        # 0: Under-voltage detected
        # 1: Arm frequency capped
        # 2: Currently throttled
        # 3: Soft temperature limit active
        # 16: Under-voltage has occurred
        # 17: Arm frequency capped has occurred
        # 18: Throttling has occurred
        # 19: Soft temperature limit has occurred
        return {
            "raw": hex_str,
            "undervoltage": bool(flags & (1 << 0)),
            "freq_capped": bool(flags & (1 << 1)),
            "throttled": bool(flags & (1 << 2)),
            "soft_temp_limit": bool(flags & (1 << 3)),
            "undervoltage_occurred": bool(flags & (1 << 16)),
            "freq_capped_occurred": bool(flags & (1 << 17)),
            "throttled_occurred": bool(flags & (1 << 18)),
            "soft_temp_limit_occurred": bool(flags & (1 << 19)),
        }

    def _collect_vcgencmd_sbc(self) -> dict[str, Any] | None:
        """Collect Raspberry Pi SBC metrics from vcgencmd (memory split, clocks, voltages)."""
        sbc: dict[str, Any] = {}

        # Memory split (arm/gpu)
        arm_mem = self._run_command(["vcgencmd", "get_mem", "arm"])
        gpu_mem = self._run_command(["vcgencmd", "get_mem", "gpu"])
        if arm_mem:
            try:
                # Parse "arm=948M" format
                value = arm_mem.strip().split("=")[1]
                if value.endswith("M"):
                    sbc["arm_mem_b"] = int(value[:-1]) * 1024 * 1024
                elif value.endswith("G"):
                    sbc["arm_mem_b"] = int(float(value[:-1]) * 1024 * 1024 * 1024)
            except (IndexError, ValueError):
                pass
        if gpu_mem:
            try:
                value = gpu_mem.strip().split("=")[1]
                if value.endswith("M"):
                    sbc["gpu_mem_b"] = int(value[:-1]) * 1024 * 1024
                elif value.endswith("G"):
                    sbc["gpu_mem_b"] = int(float(value[:-1]) * 1024 * 1024 * 1024)
            except (IndexError, ValueError):
                pass

        # Clock frequencies
        clocks: dict[str, int] = {}
        clock_names = [("arm", "arm_hz"), ("core", "core_hz"), ("h264", "h264_hz"),
                       ("v3d", "v3d_hz"), ("emmc", "emmc_hz")]
        for vcg_name, schema_name in clock_names:
            output = self._run_command(["vcgencmd", "measure_clock", vcg_name])
            if output:
                try:
                    # Parse "frequency(48)=1800000000" format
                    freq_str = output.strip().split("=")[1]
                    clocks[schema_name] = int(freq_str)
                except (IndexError, ValueError):
                    pass
        if clocks:
            sbc["clocks"] = clocks

        # Voltages
        voltages: dict[str, float] = {}
        voltage_names = [("core", "core_v"), ("sdram_c", "sdram_c_v"),
                         ("sdram_i", "sdram_i_v"), ("sdram_p", "sdram_p_v")]
        for vcg_name, schema_name in voltage_names:
            output = self._run_command(["vcgencmd", "measure_volts", vcg_name])
            if output:
                try:
                    # Parse "volt=0.9460V" format
                    volt_str = output.strip().split("=")[1]
                    if volt_str.endswith("V"):
                        voltages[schema_name] = float(volt_str[:-1])
                except (IndexError, ValueError):
                    pass
        if voltages:
            sbc["voltages"] = voltages

        return sbc if sbc else None

    def _collect_cpu_freq_scaling(self) -> dict[str, Any] | None:
        """Collect CPU frequency scaling info from Linux sysfs."""
        result: dict[str, Any] = {}
        cpu0_path = "/sys/devices/system/cpu/cpu0/cpufreq"

        # Governor
        governor = self._read_file(f"{cpu0_path}/scaling_governor")
        if governor:
            result["governor"] = governor.strip()

        # Min/max frequencies (in kHz in sysfs, convert to Hz)
        min_freq = self._read_file(f"{cpu0_path}/scaling_min_freq")
        if min_freq:
            try:
                result["freq_min_hz"] = int(min_freq.strip()) * 1000
            except ValueError:
                pass

        max_freq = self._read_file(f"{cpu0_path}/scaling_max_freq")
        if max_freq:
            try:
                result["freq_max_hz"] = int(max_freq.strip()) * 1000
            except ValueError:
                pass

        return result if result else None

    def _parse_meminfo(self) -> dict[str, int] | None:
        """Parse /proc/meminfo and return values in bytes."""
        content = self._read_file("/proc/meminfo")
        if not content:
            return None

        result: dict[str, int] = {}
        for line in content.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                try:
                    # Values are in kB
                    value_kb = int(parts[1])
                    result[key] = value_kb * 1024
                except ValueError:
                    pass
        return result

    def _collect_iface_link_details(self, iface: str) -> dict[str, Any] | None:
        """Collect network interface link details from Linux sysfs."""
        result: dict[str, Any] = {}
        sysfs_path = f"/sys/class/net/{iface}"

        # Link speed (Mbps)
        speed = self._read_file(f"{sysfs_path}/speed")
        if speed:
            try:
                speed_val = int(speed.strip())
                if speed_val > 0:  # -1 means unknown
                    result["link_speed_mbps"] = speed_val
            except ValueError:
                pass

        # Duplex mode
        duplex = self._read_file(f"{sysfs_path}/duplex")
        if duplex:
            duplex_val = duplex.strip().lower()
            if duplex_val in ("full", "half"):
                result["duplex"] = duplex_val
            elif duplex_val:
                result["duplex"] = "unknown"

        # MTU
        mtu = self._read_file(f"{sysfs_path}/mtu")
        if mtu:
            try:
                result["mtu"] = int(mtu.strip())
            except ValueError:
                pass

        # Carrier (link up/down)
        carrier = self._read_file(f"{sysfs_path}/carrier")
        if carrier:
            try:
                result["carrier"] = bool(int(carrier.strip()))
            except ValueError:
                pass

        return result if result else None

    def _collect_wifi_metrics(self, iface: str) -> dict[str, Any] | None:
        """Collect WiFi metrics for a wireless interface using iwconfig."""
        # First check if this is a wireless interface
        wireless_path = f"/sys/class/net/{iface}/wireless"
        if not Path(wireless_path).exists():
            return None

        result: dict[str, Any] = {}

        # Try iwconfig for detailed WiFi info
        output = self._run_command(["iwconfig", iface])
        if output:
            lines = output.replace("\n          ", " ").split("\n")
            for line in lines:
                # ESSID
                if "ESSID:" in line:
                    try:
                        essid = line.split('ESSID:"')[1].split('"')[0]
                        if essid and essid != "off/any":
                            result["essid"] = essid
                    except (IndexError, ValueError):
                        pass

                # Access Point
                if "Access Point:" in line:
                    try:
                        ap = line.split("Access Point:")[1].strip().split()[0]
                        if ap and ap != "Not-Associated":
                            result["access_point"] = ap
                    except (IndexError, ValueError):
                        pass

                # Mode
                if "Mode:" in line:
                    try:
                        mode = line.split("Mode:")[1].strip().split()[0]
                        result["mode"] = mode
                    except (IndexError, ValueError):
                        pass

                # Frequency
                if "Frequency:" in line:
                    try:
                        freq_str = line.split("Frequency:")[1].strip().split()[0]
                        # Convert GHz to MHz
                        if "GHz" in line:
                            result["frequency_mhz"] = float(freq_str) * 1000
                        else:
                            result["frequency_mhz"] = float(freq_str)
                    except (IndexError, ValueError):
                        pass

                # TX Power
                if "Tx-Power=" in line:
                    try:
                        tx_str = line.split("Tx-Power=")[1].strip().split()[0]
                        result["tx_power_dbm"] = int(tx_str)
                    except (IndexError, ValueError):
                        pass

                # Bit Rate
                if "Bit Rate=" in line:
                    try:
                        rate_str = line.split("Bit Rate=")[1].strip().split()[0]
                        result["bitrate_mbps"] = float(rate_str)
                    except (IndexError, ValueError):
                        pass

                # Signal level
                if "Signal level=" in line:
                    try:
                        sig_str = line.split("Signal level=")[1].strip().split()[0]
                        result["signal_dbm"] = int(sig_str)
                    except (IndexError, ValueError):
                        pass

                # Link Quality
                if "Link Quality=" in line:
                    try:
                        qual_str = line.split("Link Quality=")[1].strip().split()[0]
                        if "/" in qual_str:
                            num, denom = qual_str.split("/")
                            result["signal_pct"] = (int(num) / int(denom)) * 100
                    except (IndexError, ValueError):
                        pass

        return result if result else None

    def _collect_iface_link_details_windows(self, iface: str) -> dict[str, Any] | None:
        """Collect network interface link details on Windows using Get-NetAdapter."""
        # Get-NetAdapter returns adapter details including speed, status, and MTU
        ps_cmd = f"""
        $adapter = Get-NetAdapter -Name '{iface}' -ErrorAction SilentlyContinue
        if ($adapter) {{
            $props = @{{
                'LinkSpeed' = $adapter.LinkSpeed
                'MediaConnectionState' = $adapter.MediaConnectionState
                'FullDuplex' = $adapter.FullDuplex
                'Status' = $adapter.Status
                'InterfaceDescription' = $adapter.InterfaceDescription
            }}
            # Get MTU from Get-NetIPInterface
            $ipif = Get-NetIPInterface -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue
            if ($ipif) {{
                $props['Mtu'] = $ipif.NlMtu
            }}
            $props | ConvertTo-Json
        }}
        """
        output = self._run_command(["powershell", "-NoProfile", "-Command", ps_cmd])
        if not output:
            return None

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return None

        result: dict[str, Any] = {}

        # Parse link speed (e.g., "1 Gbps", "100 Mbps")
        link_speed = data.get("LinkSpeed", "")
        if link_speed:
            try:
                # Parse speed string like "1 Gbps" or "100 Mbps"
                parts = link_speed.split()
                if len(parts) >= 2:
                    speed_val = float(parts[0])
                    unit = parts[1].lower()
                    if "gbps" in unit:
                        result["link_speed_mbps"] = int(speed_val * 1000)
                    elif "mbps" in unit:
                        result["link_speed_mbps"] = int(speed_val)
            except (ValueError, IndexError):
                pass

        # Duplex
        full_duplex = data.get("FullDuplex")
        if full_duplex is not None:
            result["duplex"] = "full" if full_duplex else "half"

        # MTU
        mtu = data.get("Mtu")
        if mtu:
            try:
                result["mtu"] = int(mtu)
            except (ValueError, TypeError):
                pass

        # Carrier (link up/down)
        status = data.get("Status", "")
        if status:
            result["carrier"] = status.lower() == "up"

        # Driver (interface description often contains driver/adapter info)
        desc = data.get("InterfaceDescription")
        if desc:
            result["driver"] = desc

        return result if result else None

    def _collect_wifi_metrics_windows(self, iface: str) -> dict[str, Any] | None:
        """Collect WiFi metrics for a wireless interface on Windows using netsh."""
        # Check if this interface is the WiFi adapter by checking its name
        # Windows WiFi adapters typically have "Wi-Fi" or "Wireless" in the name
        iface_lower = iface.lower()
        if not ("wi-fi" in iface_lower or "wifi" in iface_lower or "wireless" in iface_lower):
            return None

        # Use netsh to get WiFi interface details
        output = self._run_command(["netsh", "wlan", "show", "interfaces"])
        if not output:
            return None

        result: dict[str, Any] = {}

        # Parse netsh output - it shows all wireless interfaces
        # We need to find the section for our interface
        lines = output.split("\n")
        in_our_interface = False

        for line in lines:
            line = line.strip()

            # Check if we're entering the section for our interface
            if line.startswith("Name") and ":" in line:
                current_name = line.split(":", 1)[1].strip()
                in_our_interface = current_name.lower() == iface_lower

            if not in_our_interface:
                continue

            # Parse fields
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "ssid":
                result["essid"] = value
            elif key == "bssid":
                result["access_point"] = value
            elif key == "network type":
                result["mode"] = value
            elif key == "channel":
                try:
                    result["channel"] = int(value)
                except ValueError:
                    pass
            elif key == "receive rate (mbps)":
                try:
                    result["rx_rate_mbps"] = float(value)
                except ValueError:
                    pass
            elif key == "transmit rate (mbps)":
                try:
                    result["tx_rate_mbps"] = float(value)
                    result["bitrate_mbps"] = float(value)  # Also set bitrate for compatibility
                except ValueError:
                    pass
            elif key == "signal":
                # Signal is reported as percentage like "98%"
                try:
                    sig_pct = int(value.rstrip("%"))
                    result["signal_pct"] = sig_pct
                    # Approximate dBm from percentage
                    # Common approximation: dBm = (signal% / 2) - 100
                    result["signal_dbm"] = int((sig_pct / 2) - 100)
                except ValueError:
                    pass
            elif key == "radio type":
                result["radio_type"] = value  # e.g., "802.11ax"
            elif key == "authentication":
                result["authentication"] = value
            elif key == "cipher":
                result["cipher"] = value

        return result if result else None

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
            drive: dict[str, Any] = {}
            for sensor in sensors:
                category = sensor["category"]
                label = sensor["name"].lower()
                value = sensor["value"]
                if category == "temperature":
                    drive["temp_c"] = float(value)
                elif category == "load":
                    if "read activity" in label:
                        drive["read_activity_pct"] = float(value)
                    elif "write activity" in label:
                        drive["write_activity_pct"] = float(value)
                elif category == "throughput":
                    if "read rate" in label:
                        drive["read_rate_bps"] = int(value)
                    elif "write rate" in label:
                        drive["write_rate_bps"] = int(value)
                elif category == "data":
                    if "data read" in label:
                        drive["total_read_b"] = int(value)
                    elif "data written" in label:
                        drive["total_write_b"] = int(value)
            if drive:
                parsed_drives[name] = drive

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
            if "cpu" in lowered:
                return "cpu"
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
                    elif sensor_type == "smalldata":
                        # GPU memory usage - LHM reports in MB or GB
                        # Try to determine if it's used or total based on label
                        if "used" in label or "gpu memory" in label:
                            # LHM reports in MB, convert to bytes
                            gpu.setdefault("memory", {})["used_b"] = int(value * 1024 * 1024)
                        elif "free" in label or "available" in label:
                            gpu.setdefault("memory", {})["available_b"] = int(value * 1024 * 1024)
                        elif "total" in label:
                            gpu.setdefault("memory", {})["total_b"] = int(value * 1024 * 1024)
                        elif "shared" in label:
                            gpu.setdefault("memory_shared", {})["used_b"] = int(value * 1024 * 1024)
                        elif "d3d" in label:
                            # D3D dedicated memory
                            gpu.setdefault("memory_d3d", {})["used_b"] = int(value * 1024 * 1024)
                if value is not None and next_type == "cpu":
                    label = node_text.lower()
                    if sensor_type == "temperature":
                        # Handle various CPU temperature sensor names
                        if any(t in label for t in ["tctl", "tdie", "cpu temp", "package", "core temp"]):
                            cpu_entry["temp_c"] = float(value)
                        elif "core #" in label:
                            core_index = self._parse_core_index(label)
                            if core_index is not None:
                                cpu_entry["cores"].setdefault(core_index, {})["temp_c"] = float(value)
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
                    elif sensor_type == "power":
                        if "core #" in label:
                            core_index = self._parse_core_index(label)
                            if core_index is not None:
                                cpu_entry["cores"].setdefault(core_index, {})[
                                    "power_w"
                                ] = float(value)
                        elif any(t in label for t in ["package", "cpu package", "total"]):
                            cpu_entry["power_w"] = float(value)
                    elif sensor_type == "clock":
                        if "core #" in label:
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
        if cpu_entry.get("temp_c") or cpu_entry.get("voltage_core_v") or cpu_entry.get("power_w") or cpu_entry["cores"]:
            parsed_cpu = cpu_entry

        return {
            "motherboard_sensors": motherboard_sensors,
            "gpus": parsed_gpus,
            "batteries": batteries,
            "drives": drive_map,
            "cpu": parsed_cpu,
            "motherboard_info": motherboard_info if motherboard_info else None,
        }
