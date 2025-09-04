from __future__ import annotations

"""Minimal execution bridge routing actions to gateways."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json
import time

from .gateway_base import GatewayBase
from .paper_gateway import PaperGateway
from .ccxt_adapter import CCXTAdapter


def _atomic_write(path: Path, line: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(line + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass
class ExecutionBridge:
    mode: str
    perf_dir: Path
    gateway: GatewayBase

    @classmethod
    def for_mode(cls, mode: str, symbol: str, perf_dir: Path) -> "ExecutionBridge":
        if mode == "paper":
            gw: GatewayBase = PaperGateway(symbol)
        elif mode == "sandbox":
            gw = CCXTAdapter(symbol)
        else:
            gw = PaperGateway(symbol)
        return cls(mode=mode, perf_dir=perf_dir, gateway=gw)

    def place(self, side: str, qty: float, price: float | None = None) -> Dict[str, Any]:
        order = self.gateway.place_order(side, qty, price)
        line = json.dumps({"ts": time.time(), "order": order})
        _atomic_write(self.perf_dir / "orders.jsonl", line)
        return order
