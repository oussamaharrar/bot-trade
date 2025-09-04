import json
import os
import threading
from dataclasses import dataclass
from queue import Queue
from typing import Optional, Dict, Any

import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


@dataclass
class WatchEvent:
    kind: str
    path: str
    data: Dict[str, Any]


class _Handler(FileSystemEventHandler):
    def __init__(self, queue: Queue):
        self.queue = queue

    def on_modified(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        path = event.src_path
        fname = os.path.basename(path)
        if fname == "metrics.csv":
            try:
                df = pd.read_csv(path)
                last = df.tail(1).to_dict("records")[0] if not df.empty else {}
                self.queue.put(WatchEvent("metrics", path, last))
            except Exception as exc:  # pragma: no cover - logged via queue
                self.queue.put(WatchEvent("metrics_error", path, {"error": str(exc)}))
        elif fname == "summary.json":
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self.queue.put(WatchEvent("summary", path, data))
            except Exception as exc:  # pragma: no cover
                self.queue.put(WatchEvent("summary_error", path, {"error": str(exc)}))
        elif fname == "risk_flags.jsonl":
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()[-5:]
                self.queue.put(WatchEvent("risk_flags", path, {"lines": lines}))
            except Exception as exc:  # pragma: no cover
                self.queue.put(WatchEvent("risk_flags_error", path, {"error": str(exc)}))


class ResultsWatcher:
    """Watch results directory for updated artefacts."""

    def __init__(self, run_dir: str, queue: Optional[Queue] = None) -> None:
        self.run_dir = run_dir
        self.queue = queue or Queue()
        self.observer: Optional[Observer] = None
        self.handler = _Handler(self.queue)

    def start(self) -> None:
        self.observer = Observer()
        self.observer.schedule(self.handler, self.run_dir, recursive=False)
        self.observer.start()

    def stop(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2)
            self.observer = None

    def join(self) -> None:
        if self.observer:
            self.observer.join()
