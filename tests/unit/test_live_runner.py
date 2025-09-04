import sys
import time
from pathlib import Path

from bot_trade.runners import live_dry_run


class OneTickFeed:
    def __init__(self, *args, **kwargs):
        self.interval = 0.01
        self.last_price = None

    def stream(self, symbol, on_tick):
        self.last_price = 100.0
        on_tick(100.0)


class NoPriceFeed:
    def __init__(self, *args, **kwargs):
        self.interval = 0.1
        self.last_price = None

    def stream(self, symbol, on_tick):
        pass


def test_model_optional_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live_dry_run, "LiveFeed", OneTickFeed)
    monkeypatch.setattr(
        live_dry_run, "_load_config", lambda _: {"binance": {"rest": "", "price_path": ""}}
    )
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def time_gen():
        t = 0.0
        while True:
            t += 1.0
            yield t

    tg = time_gen()
    monkeypatch.setattr(time, "time", lambda: next(tg))

    argv = [
        "prog",
        "--exchange",
        "binance",
        "--symbol",
        "BTCUSDT",
        "--frame",
        "1m",
        "--duration",
        "1",
        "--model-optional",
        "--bootstrap-price",
        "27000",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    live_dry_run.main()

    run_dir = next(Path("results").rglob("summary.json")).parent
    log = (run_dir / "logs" / "run.log").read_text()
    assert "[LIVE] tick=" in log
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "risk_flags.jsonl").exists()


def test_hold_policy_after_bad_polls(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live_dry_run, "LiveFeed", NoPriceFeed)
    monkeypatch.setattr(
        live_dry_run, "_load_config", lambda _: {"binance": {"rest": "", "price_path": ""}}
    )
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def time_gen():
        t = 0.0
        while True:
            t += 0.2
            yield t

    tg = time_gen()
    monkeypatch.setattr(time, "time", lambda: next(tg))

    argv = [
        "prog",
        "--exchange",
        "binance",
        "--symbol",
        "BTCUSDT",
        "--frame",
        "1m",
        "--duration",
        "2",
        "--model-optional",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    live_dry_run.main()
    out = capsys.readouterr().out
    assert "no valid price" in out
    run_dir = next(Path("results").rglob("summary.json")).parent
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "risk_flags.jsonl").exists()
