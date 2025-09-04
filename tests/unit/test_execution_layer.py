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
    assert res.fills and res.fills[0].min_fee_applied


def test_maker_fee_and_min_fee():
    cfg = {"fees": {"maker_bps": 1, "taker_bps": 5, "min_fee": 0.5}}
    layer = ExecutionLayer(cfg, seed=1)
    order = Order(id="2", side="sell", qty=2, price=50.0, ts=0.0, is_maker=True)
    res = layer.apply(order, {"price": 50.0})
    assert res.status == "filled"
    # maker fee 1 bps -> 0.01% * value
    assert res.fees == max(50.0 * 2 * 0.0001, 0.5)
    assert res.fills and res.fills[0].is_maker
