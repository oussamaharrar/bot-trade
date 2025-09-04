import sys
from pathlib import Path

import pytest
import sys

from bot_trade.runners import live_dry_run


class NoPriceFeed:
    def __init__(self, *args, **kwargs):
        self.interval = 0.01
        self.last_price = None

    def stream(self, symbol, on_tick):
        # never emits price
        pass


def test_no_valid_price_for_10(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(live_dry_run, "LiveFeed", NoPriceFeed)
    monkeypatch.setattr(
        live_dry_run, "_load_config", lambda _: {"binance": {"rest": "", "price_path": ""}}
    )
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
        "--config",
        "cfg",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    live_dry_run.main()
    out = capsys.readouterr().out
    assert "no valid price" in out
    run_dir = next(Path("results").rglob("summary.json")).parent
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "risk_flags.jsonl").exists()
