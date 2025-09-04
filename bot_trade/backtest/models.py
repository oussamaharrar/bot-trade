from dataclasses import dataclass


@dataclass(slots=True)
class Order:
    """Simplified order model used for backtests."""

    id: str
    side: str
    qty: float
    price: float
    ts: float
    is_maker: bool = False


@dataclass(slots=True)
class Fill:
    """Executed order slice."""

    side: str
    qty: float
    price: float
    fee: float
    slippage_bps: float
    latency_ms: int


@dataclass(slots=True)
class ExecutionResult:
    """Aggregate execution outcome."""

    status: str
    filled_qty: float
    avg_price: float
    fees: float
    ts: float
    slippage_bps: float
    order_id: str | None = None
    fills: list[Fill] | None = None

