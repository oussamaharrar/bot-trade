import time
from queue import Queue

from bot_trade.ui.results_watcher import ResultsWatcher


def test_results_watcher_detects_metrics(tmp_path):
    run_dir = tmp_path
    q = Queue()
    watcher = ResultsWatcher(str(run_dir), q)
    watcher.start()
    metrics = run_dir / "metrics.csv"
    metrics.write_text("a,b\n1,2\n")
    time.sleep(0.5)
    event = q.get(timeout=2)
    assert event.kind == "metrics"
    watcher.stop()
