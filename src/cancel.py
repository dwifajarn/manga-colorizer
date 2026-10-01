"""Cooperative cancellation for long-running colorization jobs.

A single, process-wide flag is enough because the CLI and GUI run one job at a
time. The GUI's worker thread calls :func:`cancel` from the Tk main thread when
the user presses **Stop**; the pipeline checks :func:`is_cancelled` between
pages and bails out cleanly.

Kept deliberately tiny and dependency-free so any module can import it without
pulling in torch/opencv.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_cancelled = False


def reset() -> None:
    """Clear the cancel flag (call this before starting a new job)."""
    global _cancelled
    with _lock:
        _cancelled = False


def cancel() -> None:
    """Request cancellation of the current job."""
    global _cancelled
    with _lock:
        _cancelled = True


def is_cancelled() -> bool:
    """True once :func:`cancel` has been called and before :func:`reset`."""
    with _lock:
        return _cancelled


class CancelledError(RuntimeError):
    """Raised when a job is aborted by the user mid-run."""
