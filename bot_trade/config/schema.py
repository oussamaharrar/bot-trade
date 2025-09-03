from __future__ import annotations

"""Pydantic config schema used by tools."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class NetConfig(BaseModel):
    layers: list[int] = Field(default_factory=lambda: [64, 64])
    activation: str = "ReLU"
    ortho_init: bool = True


class RLConfig(BaseModel):
    algorithm: str = "PPO"
    total_steps: int = 100_000


class ExecutionFees(BaseModel):
    maker_bps: float = 0.0
    taker_bps: float = 0.0


class ExecutionLimits(BaseModel):
    max_exposure: float | None = None
    min_notional: float | None = None


class ExecutionConfig(BaseModel):
    mode: str = "backtest"
    slippage: str = "fixed_bp"
    latency_ms: int = 0
    partial_fills: bool = True
    fees: ExecutionFees = ExecutionFees()
    limits: ExecutionLimits = ExecutionLimits()


class GatewayConfig(BaseModel):
    provider: str = "ccxt"
    sandbox: bool = True


class Config(BaseModel):
    rl: RLConfig = RLConfig()
    net: NetConfig = NetConfig()
    execution: ExecutionConfig = ExecutionConfig()
    gateway: GatewayConfig = GatewayConfig()
    extra: Dict[str, Any] = Field(default_factory=dict)
