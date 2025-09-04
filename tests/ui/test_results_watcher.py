import json
import time
from pathlib import Path
from queue import Queue

from bot_trade.ui.results_watcher import ResultsWatcher


def test_results_watcher_detects_metrics(tmp_path):
    run_dir = tmp_path
    perf = run_dir / "performance"
    perf.mkdir()
    metrics = perf / "metrics.csv"
    metrics.write_text("a,b\n", encoding="utf-8")
    summary = perf / "summary.json"
    log = tmp_path / "run.log"
    q: Queue = Queue()
    watcher = ResultsWatcher(run_dir, log, q, poll_sec=0.1)
    watcher.start()
    metrics.write_text("a,b\n1,2\n", encoding="utf-8")
    summary.write_text(json.dumps({"x":1}), encoding="utf-8")
    with log.open("w", encoding="utf-8") as fh:
        fh.write("[CHARTS] dir=/tmp images=5\n")
        fh.write("[POSTRUN] ok\n")
    time.sleep(0.5)
    watcher.stop()
    events = [q.get_nowait() for _ in range(q.qsize())]
    types = {e["event"] for e in events}
    assert {"metric","summary","charts","postrun"} <= types
