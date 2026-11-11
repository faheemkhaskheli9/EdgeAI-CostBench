"""System metadata and optional sampled process memory (never GPU memory)."""

from __future__ import annotations

import os
import platform
import threading


def system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        import psutil
        info["ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        info["ram_bytes"] = None
    return info


class ProcessMemorySampler:
    """Best-effort sampled peak RSS, not native or GPU allocation high-water mark."""

    def __init__(self, interval_s: float = 0.01):
        self.interval_s = interval_s
        self.peak_rss_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None

    def _sample(self) -> None:
        if self._process is not None:
            try:
                rss = self._process.memory_info().rss
                self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
            except (OSError, RuntimeError):
                pass

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        try:
            import psutil
            self._process = psutil.Process()
        except ImportError:
            return self
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._sample()
