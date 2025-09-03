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


class Config(BaseModel):
    rl: RLConfig = RLConfig()
    net: NetConfig = NetConfig()
    extra: Dict[str, Any] = Field(default_factory=dict)
