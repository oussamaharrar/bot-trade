import os
import pytest

from bot_trade.gateways.exchanges.binance_testnet import BinanceTestnet, GatewayError as BinanceError
from bot_trade.utils.rate_limit import RateLimiter


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "test")


def test_missing_env(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    with pytest.raises(BinanceError):
        BinanceTestnet("url", 5000, RateLimiter(1))


def test_signature(monkeypatch, keys):
    bt = BinanceTestnet("url", 5000, RateLimiter(1))
    params = {"symbol": "LTCBTC", "side": "BUY", "timestamp": "1499827319559"}
    signed = bt._sign(params.copy())
    assert (
        signed["signature"]
        == "16254abbf1601f5b8a39334ca92df4d264181531d1dc931a02e2f0d4e458043e"
    )
