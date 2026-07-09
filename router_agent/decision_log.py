"""Append-only JSONL decision log.

Local-only instrumentation (never sent to the harness): this log is the sole
visibility we have into our own leaderboard-proxy performance, and it is the
training data for the calibration step (03_ARCHITECTURE, Calibration
Methodology). Build order in the plan puts this before the router itself.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .router import Decision


class DecisionLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # harness mode logs from worker threads

    def log(self, decision: Decision) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(decision.to_json() + "\n")
