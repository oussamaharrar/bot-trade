import time
from pathlib import Path
from queue import Queue

from bot_trade.ui import runner


def test_runner_start_stop(tmp_path):
    log_queue = Queue()
    log_path = tmp_path / "echo.log"
    pid = runner.start_command(
        ["python", "-c", "import time,sys; print('hello'); sys.stdout.flush(); time.sleep(1)"],
        run_id="echo",
        tee_path=str(log_path),
        log_queue=log_queue,
    )
    time.sleep(0.2)
    assert log_path.exists()
    line = log_queue.get(timeout=2)
    assert "hello" in line["line"]
    res = runner.stop_process_tree(pid)
    assert res["stopped"]
