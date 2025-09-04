from __future__ import annotations

"""Watch run artifacts and emit structured events."""

import csv
import json
import os
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable


class ResultsWatcher:
    def __init__(self, run_dir: Path, log_path: Path, out_queue: Queue, poll_sec: float = 0.5) -> None:
        self.run_dir = run_dir
        self.log_path = log_path
        self.queue = out_queue
        self.poll_sec = poll_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._metrics_pos = 0
        self._log_pos = 0
        self._summary_mtime = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        metrics_path = self.run_dir / "performance" / "metrics.csv"
        summary_path = self.run_dir / "performance" / "summary.json"
        while not self._stop.is_set():
            try:
                if metrics_path.exists():
                    self._check_metrics(metrics_path)
                if summary_path.exists():
                    self._check_summary(summary_path)
                if self.log_path.exists():
                    self._check_log(self.log_path)
            except Exception:
                pass
            time.sleep(self.poll_sec)

    def _check_metrics(self, path: Path) -> None:
        size = path.stat().st_size
        if size <= self._metrics_pos:
            return
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(self._metrics_pos)
            reader = csv.DictReader(fh)
            for row in reader:
                self.queue.put({"event": "metric", "data": row})
            self._metrics_pos = fh.tell()

    def _check_summary(self, path: Path) -> None:
        mtime = path.stat().st_mtime
        if mtime <= self._summary_mtime:
            return
        self._summary_mtime = mtime
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.queue.put({"event": "summary", "data": data})
        except Exception:
            pass

    def _check_log(self, path: Path) -> None:
        size = path.stat().st_size
        if size <= self._log_pos:
            return
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(self._log_pos)
            for line in fh:
                line = line.strip()
                if line.startswith("[CHARTS] "):
                    self.queue.put({"event": "charts", "line": line})
                elif line.startswith("[POSTRUN] "):
                    self.queue.put({"event": "postrun", "line": line})
            self._log_pos = fh.tell()


__all__ = ["ResultsWatcher"]
