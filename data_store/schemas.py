from __future__ import annotations

"""Pydantic models for bars and ticks."""

from datetime import datetime
from pydantic import BaseModel, Field


class Bar(BaseModel):
    datetime: datetime = Field(..., description="UTC timestamp")
    open: float
    high: float
    low: float
    close: float
    volume: float


class Tick(BaseModel):
    datetime: datetime = Field(..., description="UTC timestamp")
    price: float
    volume: float

