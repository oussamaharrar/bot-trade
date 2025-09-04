import time
from typing import List

import pytest
import requests

from bot_trade.data.live_feed import LiveFeed


class DummyResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_bad_price_sequence(monkeypatch):
    seq = [
        {"price": 10},
        {"price": None},
        {"price": float("nan")},
    ]
    calls = {"i": 0}

    def fake_get(url, params=None):
        payload = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        return DummyResp(payload)

    sleeps: List[float] = []

    def fake_sleep(t):
        sleeps.append(t)
        if len(sleeps) >= 3:
            raise RuntimeError

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    feed = LiveFeed(None, "http://test", interval=0.1, alpha=1.0)
    monkeypatch.setattr(feed.limiter, "acquire", lambda *a, **k: None)
    prices: List[float] = []

    def on_tick(p: float) -> None:
        prices.append(p)

    with pytest.raises(RuntimeError):
        feed._http_loop("BTC", on_tick)

    assert prices == [10, 10, 10]
    assert sleeps[1] < sleeps[2]


def test_ewma_smoothing(monkeypatch):
    seq = [{"price": 10}, {"price": 11}, {"price": 13}]
    calls = {"i": 0}

    def fake_get(url, params=None):
        payload = seq[calls["i"]]
        calls["i"] += 1
        return DummyResp(payload)

    def fake_sleep(t):
        if calls["i"] >= 3:
            raise RuntimeError

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    feed = LiveFeed(None, "http://test", interval=0.1, alpha=0.3)
    monkeypatch.setattr(feed.limiter, "acquire", lambda *a, **k: None)
    prices: List[float] = []

    def on_tick(p: float) -> None:
        prices.append(p)

    with pytest.raises(RuntimeError):
        feed._http_loop("BTC", on_tick)

    assert prices == pytest.approx([10, 10.3, 11.11], rel=1e-2)
