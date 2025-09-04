import os
import pytest

from bot_trade.gateways.exchanges.binance_testnet import BinanceTestnet
from bot_trade.gateways.errors import GatewayError
from bot_trade.utils.rate_limit import RateLimiter


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "KEY12345")
    monkeypatch.setenv("BINANCE_API_SECRET", "SECRET12345")


def test_missing_env(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    with pytest.raises(GatewayError):
        BinanceTestnet("url", 5000, RateLimiter(1), {})


def test_signature(monkeypatch, keys):
    bt = BinanceTestnet("url", 5000, RateLimiter(1), {})
    params = {"symbol": "LTCBTC", "side": "BUY", "timestamp": "1499827319559"}
    signed = bt._sign(params.copy())
    assert (
        signed["signature"]
        == "cbb31cc794b379e5f51438435da780bc61e69dea83409781a2cac9b227b1691f"
    )
