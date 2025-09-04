import time
from bot_trade.utils.rate_limit import RateLimiter, retry


def test_rate_limiter_budget():
    rl = RateLimiter(capacity=2, refill_time=1)
    start = time.time()
    rl.acquire(1)
    rl.acquire(1)
    rl.acquire(1)
    # third acquire should block roughly one second
    assert time.time() - start >= 0.5


def test_retry_on_code(capfd):
    calls = {"n": 0}

    def func():
        calls["n"] += 1
        if calls["n"] < 2:
            return {"status": 429}
        return {"status": 200}

    resp = retry(func, is_retryable=lambda r: r["status"] == 429, base_delay=0.1)
    assert resp["status"] == 200
    captured = capfd.readouterr().out
    assert "RLIMIT_SLEEP" in captured
