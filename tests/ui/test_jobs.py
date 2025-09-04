import sys
import time

from bot_trade.tools import runctx


def test_jobs_start_list_stop():
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    h1 = runctx.start(cmd)
    h2 = runctx.start(cmd)
    time.sleep(0.2)
    jobs = runctx.list_jobs()
    ids = {j["id"] for j in jobs}
    assert h1.id in ids and h2.id in ids
    runctx.stop(h1.id)
    runctx.stop(h2.id)
