import time

import pytest
import requests

from bot_trade.data.live_feed import LiveFeed


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_smoothing_and_last_price_fallback(monkeypatch):
    seq = iter([
        {"price": 100.0},
        {"price": 110.0},
        {"price": 0.0},
    ])

    def fake_get(url, params=None):
        return DummyResp(next(seq))

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    feed = LiveFeed(None, "http://test", interval=0.01, alpha=0.3)
    monkeypatch.setattr(feed.limiter, "acquire", lambda *a, **k: None)

    prices: list[float] = []

    def on_tick(p: float) -> None:
        prices.append(p)
        if len(prices) >= 3:
            raise SystemExit

    with pytest.raises(SystemExit):
        feed._http_loop("BTCUSDT", on_tick)

    assert prices[0] == pytest.approx(100.0)
    assert prices[1] == pytest.approx(103.0, rel=1e-3)
    assert prices[2] == pytest.approx(105.1, rel=1e-3)


def test_capped_backoff(monkeypatch, capsys):
    def fake_get(url, params=None):
        return DummyResp({})

    sleep_calls = {"n": 0}

    def fake_sleep(s):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 7:
            raise SystemExit

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    feed = LiveFeed(None, "http://test", interval=0.01)
    monkeypatch.setattr(feed.limiter, "acquire", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        feed._http_loop("BTCUSDT", lambda p: None)

    out = capsys.readouterr().out
    assert "bad_price" in out
    assert "backoff_ms=2000" in out
