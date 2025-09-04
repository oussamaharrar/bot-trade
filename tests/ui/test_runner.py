import sys
import time
from pathlib import Path

from bot_trade.ui import runner


def test_runner_start_stop(tmp_path):
    log = tmp_path / "tee.log"
    script = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        "print('hello');sys.stdout.flush();time.sleep(30)"
    )
    handle = runner.start_command([sys.executable, "-c", script], tee_path=log, metadata={})
    assert handle.pid > 0
    time.sleep(1)
    rc = runner.stop_process_tree(handle, grace_sec=1)
    assert handle.process.poll() is not None
    assert log.read_text(encoding="utf-8").strip() != ""
