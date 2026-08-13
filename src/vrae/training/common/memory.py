from __future__ import annotations

import ctypes
import gc
import os
import sys
from pathlib import Path

_GIB = float(1024**3)
_M_TRIM_THRESHOLD = -1
_M_ARENA_MAX = -8
_LIBC: object | bool | None = None


def _load_libc():
    global _LIBC
    if _LIBC is None and sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
            libc.mallopt.restype = ctypes.c_int
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            _LIBC = libc
        except (AttributeError, OSError):
            _LIBC = False
    return None if _LIBC is False else _LIBC


def configure_glibc_allocator(
    *, arena_max: int = 2, trim_threshold_bytes: int = 128 * 1024**2
) -> bool:
    """Bound native allocator growth for long-lived decode threads."""

    libc = _load_libc()
    if libc is None:
        return False
    configured = True
    if int(arena_max) > 0:
        configured = bool(libc.mallopt(_M_ARENA_MAX, int(arena_max))) and configured
    if int(trim_threshold_bytes) > 0:
        configured = bool(libc.mallopt(_M_TRIM_THRESHOLD, int(trim_threshold_bytes))) and configured
    return configured


def trim_process_heap(*, collect_python: bool = False) -> bool:
    if collect_python:
        gc.collect()
    libc = _load_libc()
    return bool(libc is not None and libc.malloc_trim(0))


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _read_keyed_int(path: Path, key: str) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] == key:
            try:
                return int(fields[1])
            except ValueError:
                return None
    return None


def host_memory_metrics() -> dict[str, float]:
    """Return cheap process, host, and cgroup memory metrics."""

    metrics: dict[str, float] = {}
    system_root = Path(os.sep)
    rss_kib = _read_keyed_int(system_root / "proc/self/status", "VmRSS:")
    if rss_kib is not None:
        metrics["memory/host_process_rss_gb"] = rss_kib * 1024 / _GIB

    meminfo = system_root / "proc/meminfo"
    total_kib = _read_keyed_int(meminfo, "MemTotal:")
    available_kib = _read_keyed_int(meminfo, "MemAvailable:")
    active_anon_kib = _read_keyed_int(meminfo, "Active(anon):")
    inactive_anon_kib = _read_keyed_int(meminfo, "Inactive(anon):")
    if total_kib is not None:
        metrics["memory/host_total_gb"] = total_kib * 1024 / _GIB
    if available_kib is not None:
        metrics["memory/host_available_gb"] = available_kib * 1024 / _GIB
    if active_anon_kib is not None and inactive_anon_kib is not None:
        metrics["memory/host_anon_gb"] = (active_anon_kib + inactive_anon_kib) * 1024 / _GIB

    system_root = Path(os.sep)
    v1_root = system_root / "sys/fs/cgroup/memory"
    usage = _read_int(v1_root / "memory.usage_in_bytes")
    limit = _read_int(v1_root / "memory.limit_in_bytes")
    peak = _read_int(v1_root / "memory.max_usage_in_bytes")
    cache = _read_keyed_int(v1_root / "memory.stat", "total_cache")
    anon = _read_keyed_int(v1_root / "memory.stat", "total_rss")
    oom_kill = _read_keyed_int(v1_root / "memory.oom_control", "oom_kill")

    if usage is None:
        v2_root = system_root / "sys/fs/cgroup"
        usage = _read_int(v2_root / "memory.current")
        limit = _read_int(v2_root / "memory.max")
        peak = _read_int(v2_root / "memory.peak")
        cache = _read_keyed_int(v2_root / "memory.stat", "file")
        anon = _read_keyed_int(v2_root / "memory.stat", "anon")
        oom_kill = _read_keyed_int(v2_root / "memory.events", "oom_kill")

    if usage is not None:
        metrics["memory/cgroup_used_gb"] = usage / _GIB
    if limit is not None and limit < 1 << 60:
        metrics["memory/cgroup_limit_gb"] = limit / _GIB
        if usage is not None and limit > 0:
            metrics["memory/cgroup_used_pct"] = 100.0 * usage / limit
    if peak is not None:
        metrics["memory/cgroup_peak_gb"] = peak / _GIB
    if cache is not None:
        metrics["memory/cgroup_cache_gb"] = cache / _GIB
    if anon is not None:
        metrics["memory/cgroup_anon_gb"] = anon / _GIB
    if oom_kill is not None:
        metrics["memory/cgroup_oom_kill"] = float(oom_kill)
    return metrics
