"""Raspberry Pi health readings for the dashboard (later: Discord / MQTT alerts)."""
import os
import shutil
import subprocess
import time


def _vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


# Bits of `vcgencmd get_throttled`: 0-3 = happening now, 16-19 = since boot.
THROTTLE_BITS = {0: "under-voltage", 1: "arm frequency capped", 2: "throttled", 3: "soft temperature limit"}


def read(root="/"):
    h = {}
    try:
        h["cpu_temp_c"] = round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000, 1)
    except OSError:
        h["cpu_temp_c"] = None
    raw = _vcgencmd("get_throttled").partition("=")[2]
    try:
        bits = int(raw, 16)
        h["throttle_now"] = [n for b, n in THROTTLE_BITS.items() if bits & (1 << b)]
        h["throttle_since_boot"] = [n for b, n in THROTTLE_BITS.items() if bits & (1 << (b + 16))]
    except ValueError:
        h["throttle_now"] = h["throttle_since_boot"] = None
    volt = _vcgencmd("pmic_read_adc", "EXT5V_V").partition("=")[2].rstrip("V")
    try:
        h["supply_v"] = round(float(volt), 2)
    except ValueError:
        h["supply_v"] = None
    try:
        b = open("/proc/device-tree/chosen/power/max_current", "rb").read()
        h["supply_max_ma"] = int.from_bytes(b[:4], "big")
    except OSError:
        h["supply_max_ma"] = None
    du = shutil.disk_usage(root)
    h["disk_used_pct"] = round(100 * du.used / du.total, 1)
    h["disk_free_gb"] = round(du.free / 1e9, 1)
    h["uptime_s"] = int(float(open("/proc/uptime").read().split()[0]))
    h["load_1m"] = round(os.getloadavg()[0], 2)
    h["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return h
