from bot_trade.backtest.execution_layer import ExecutionLayer
from bot_trade.backtest.models import Order


def test_execution_layer_basic():
    cfg = {
        "slippage": {"model": "fixed_bps", "bps": 10},
        "fees": {"taker_bps": 5, "min_fee": 1},
        "latency_ms": 1000,
        "partial_fills": {"fill_ratio": 0.5},
    }
    layer = ExecutionLayer(cfg, seed=42)
    order = Order(id="1", side="buy", qty=10, price=100.0, ts=0.0)
    market = {"price": 100.0}
    res = layer.apply(order, market)
    assert res.status == "partial"
    assert res.filled_qty == 5
    assert res.avg_price == 100.1  # 10 bps on buy side
    assert res.fees == 1  # min fee dominates
    assert res.ts == 1.0  # latency applied
    assert res.slippage_bps == 10
