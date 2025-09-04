from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RiskRule:
    """Base interface for runtime risk checks."""

    name: str

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return evaluation dict."""
        raise NotImplementedError


@dataclass
class LossStreakHalt(RiskRule):
    streak: int = 5

    def __init__(self, streak: int = 5):
        super().__init__("LossStreakHalt")
        self.streak = int(streak)

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ls = int(state.get("loss_streak", 0))
        triggered = ls >= self.streak and self.streak > 0
        return {
            "triggered": triggered,
            "level": "warn" if triggered else "ok",
            "reason": f"loss_streak {ls} >= {self.streak}" if triggered else "",
            "adjustments": {"freeze": True} if triggered else {},
        }


@dataclass
class DrawdownStop(RiskRule):
    dd_limit: float = -0.2
    action: str = "freeze"

    def __init__(self, dd_limit: float = -0.2, action: str = "freeze"):
        super().__init__("DrawdownStop")
        self.dd_limit = float(dd_limit)
        self.action = action

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        dd = float(state.get("drawdown", 0.0))
        triggered = dd <= self.dd_limit
        adj: Dict[str, Any] = {}
        if triggered:
            if self.action == "freeze":
                adj["freeze"] = True
            else:
                adj["max_leverage"] = 0.0
        return {
            "triggered": triggered,
            "level": "warn" if triggered else "ok",
            "reason": f"drawdown {dd:.4f} <= {self.dd_limit:.4f}" if triggered else "",
            "adjustments": adj,
        }


@dataclass
class SpreadSpike(RiskRule):
    max_spread_bp: float = 10.0
    reduce: float | None = None

    def __init__(self, max_spread_bp: float = 10.0, reduce: float | None = None):
        super().__init__("SpreadSpike")
        self.max_spread_bp = float(max_spread_bp)
        self.reduce = reduce

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sp = float(state.get("spread_bp", 0.0))
        triggered = sp > self.max_spread_bp
        adj: Dict[str, Any] = {"freeze": True}
        if triggered and self.reduce is not None:
            adj["max_position"] = float(self.reduce)
        return {
            "triggered": triggered,
            "level": "warn" if triggered else "ok",
            "reason": f"spread {sp:.2f} > {self.max_spread_bp:.2f}" if triggered else "",
            "adjustments": adj if triggered else {},
        }


@dataclass
class LiquidityDrop(RiskRule):
    min_depth: float = 0.0
    cap_position: float | None = None

    def __init__(self, min_depth: float = 0.0, cap_position: float | None = None):
        super().__init__("LiquidityDrop")
        self.min_depth = float(min_depth)
        self.cap_position = cap_position

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        depth = float(state.get("depth", 0.0) or 0.0)
        triggered = depth < self.min_depth
        adj: Dict[str, Any] = {"freeze": True}
        if triggered and self.cap_position is not None:
            adj["max_position"] = float(self.cap_position)
        return {
            "triggered": triggered,
            "level": "warn" if triggered else "ok",
            "reason": f"depth {depth} < {self.min_depth}" if triggered else "",
            "adjustments": adj if triggered else {},
        }


@dataclass
class GapRisk(RiskRule):
    gap_pct: float = 0.05
    cooldown_steps: int = 0

    def __init__(self, gap_pct: float = 0.05, cooldown_steps: int = 0):
        super().__init__("GapRisk")
        self.gap_pct = float(gap_pct)
        self.cooldown_steps = int(cooldown_steps)
        self._cooldown = 0

    def evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        price_gap = float(state.get("gap_pct", 0.0))
        triggered = price_gap > self.gap_pct
        if self._cooldown > 0:
            self._cooldown -= 1
        adj: Dict[str, Any] = {}
        if triggered:
            self._cooldown = max(self._cooldown, self.cooldown_steps)
            adj["freeze"] = True
        elif self._cooldown > 0:
            triggered = True
            adj["freeze"] = True
        return {
            "triggered": triggered,
            "level": "warn" if triggered else "ok",
            "reason": f"gap {price_gap:.4f} > {self.gap_pct:.4f}" if triggered else "",
            "adjustments": adj if triggered else {},
        }


RULES_MAP = {
    "LossStreakHalt": LossStreakHalt,
    "DrawdownStop": DrawdownStop,
    "SpreadSpike": SpreadSpike,
    "LiquidityDrop": LiquidityDrop,
    "GapRisk": GapRisk,
}

__all__ = [
    "RiskRule",
    "LossStreakHalt",
    "DrawdownStop",
    "SpreadSpike",
    "LiquidityDrop",
    "GapRisk",
    "RULES_MAP",
]
