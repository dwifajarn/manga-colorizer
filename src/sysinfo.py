"""Small system-info helpers for the GUI (free RAM, coarse estimates).

Free memory is read via ``psutil`` when available, otherwise through the
Windows API with :mod:`ctypes`. Both paths are dependency-free: ``psutil`` is
optional and the code works without it.

The estimation helpers are deliberately coarse — they exist to warn a user when
a big job (especially with upscaling) is likely to exhaust RAM, not to be
precise.
"""

from __future__ import annotations

import sys


def free_ram_bytes() -> int | None:
    """Best-effort free physical RAM in bytes, or ``None`` if unknown."""
    # 1) psutil, if installed.
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - optional dependency
        pass

    # 2) Windows: GlobalMemoryStatusEx via ctypes.
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:  # noqa: BLE001
            pass

    # 3) POSIX: /proc/meminfo (Linux).
    try:
        from pathlib import Path

        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass

    return None


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    gb = n / (1024 ** 3)
    return f"{gb:.1f} GB"


def estimate_peak_ram(
    pages: int,
    upscale: bool,
    engine: str = "mangacolv2",
) -> int:
    """Coarse estimate (bytes) of the pipeline's peak RAM use.

    Rough model: a fixed base (interpreter + model weights) plus a per-page
    working set, with upscaling the dominant variable because it works on
    large, full-resolution tiles. Values are intentionally round.
    """
    base = 1_600 * 1024 * 1024  # Python + torch + weights
    if engine == "zhang":
        base = 1_400 * 1024 * 1024
    elif engine == "comicnet":
        base = 900 * 1024 * 1024

    # Working set for a single page in flight.
    per_page = 250 * 1024 * 1024
    if upscale:
        # Real-ESRGAN x2 on a full page is memory-hungry (big float buffers).
        per_page = 2_000 * 1024 * 1024

    # Only one page is processed at a time, but keep a small allowance for
    # queued buffers / OS caches proportional to the batch.
    small = min(pages, 8) * 20 * 1024 * 1024
    return base + per_page + small


def estimate_seconds(
    pages: int,
    size: int,
    upscale: bool,
    engine: str = "mangacolv2",
) -> int:
    """Very coarse wall-clock estimate in seconds (label as approximate)."""
    if pages <= 0:
        return 0
    # Seconds per page scales roughly with (size/576)^2; upscale adds a lot.
    per = 4.0 * (size / 576.0) ** 2
    if engine == "zhang":
        per *= 0.4
    elif engine == "comicnet":
        per *= 0.6
    if upscale:
        per += 6.0 * (size / 576.0) ** 2
    return max(1, int(per * pages))
