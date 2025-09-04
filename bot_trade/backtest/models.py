from dataclasses import dataclass


@dataclass(slots=True)
class Order:
    id: str
    side: str
    qty: float
    price: float
    ts: float


@dataclass(slots=True)
class ExecutionResult:
    status: str
    filled_qty: float
    avg_price: float
    fees: float
    ts: float
    slippage_bps: float
    order_id: str | None = None
