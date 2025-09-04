
from __future__ import annotations
"""Watch run artifacts and emit structured events."""

import csv
import json
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, Optional
import re


_DEBUG_RE = re.compile(r"^\[DEBUG_EXPORT\] (.*)")
_CHARTS_RE = re.compile(r"^\[CHARTS\] dir=(?P<dir>\S+) images=(?P<img>\d+)")
_POSTRUN_RE = re.compile(r"^\[POSTRUN\] (.*)")


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
        self._last_charts: Dict[str, object] = {}

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _check_metrics(self, path: Path) -> None:
        size = path.stat().st_size
        if size <= self._metrics_pos:
            return
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(self._metrics_pos)
            reader = csv.reader(fh)
            rows = list(reader)
            row_count = len(rows)
            self._metrics_pos = fh.tell()
        self.queue.put({"event": "metric", "path": str(path), "ts": time.time(), "row_count": row_count})

    def _check_summary(self, path: Path) -> None:
        mtime = path.stat().st_mtime
        if mtime <= self._summary_mtime:
            return
        self._summary_mtime = mtime
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            event = {
                "event": "summary",
                "path": str(path),
                "sharpe": data.get("eval_sharpe") or data.get("sharpe"),
                "max_dd": data.get("eval_max_drawdown") or data.get("max_dd"),
            }
            self.queue.put(event)
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
                event = parse_log_line(line)
                if not event:
                    continue
                if event["event"] == "charts":
                    self._last_charts = {"charts_dir": event["charts_dir"], "images": event["images"]}
                    self.queue.put(event)
                elif event["event"] == "postrun":
                    event.update(self._last_charts)
                    self.queue.put(event)
                else:
                    self.queue.put(event)
            self._log_pos = fh.tell()


def parse_log_line(line: str) -> Optional[Dict[str, object]]:
    m = _DEBUG_RE.match(line)
    if m:
        kv = _parse_kv(m.group(1))
        return {"event": "debug_export", **kv}
    m = _CHARTS_RE.match(line)
    if m:
        return {"event": "charts", "charts_dir": m.group("dir"), "images": int(m.group("img"))}
    m = _POSTRUN_RE.match(line)
    if m:
        kv = _parse_kv(m.group(1))
        kv["event"] = "postrun"
        return kv
    return None


def _parse_kv(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for part in text.split():
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out


__all__ = ["ResultsWatcher", "parse_log_line"]
