from __future__ import annotations
"""Risk management helpers with circuit breakers and kill switch."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

from bot_trade.tools.atomic_io import append_jsonl, write_png


@dataclass
class RiskEvent:
    ts: float
    flag: str
    value: float


@dataclass
class RiskManager:
    limits: Dict[str, float]
    events: List[RiskEvent] = field(default_factory=list)
    killed: bool = False

    def check(self, flag: str, value: float) -> None:
        limit = self.limits.get(flag)
        if limit is None:
            return
        if abs(value) > limit:
            self.events.append(RiskEvent(0.0, flag, value))

    def breach(self, reason: str) -> None:
        if not self.killed:
            print(f"[RISK_KILL] reason={reason}")
            self.killed = True

    def export(self, path: Path) -> None:
        for ev in self.events:
            append_jsonl(path.with_suffix(".jsonl"), ev.__dict__)
        if self.events:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar([e.flag for e in self.events], [e.value for e in self.events])
            ax.set_title("risk flags")
            write_png(path.with_suffix(".png"), fig)
        else:
            write_png(path.with_suffix(".png"), plt.figure(figsize=(6, 4)))


RiskRule = Callable[[Dict, float], bool]
RISK_RULES: Dict[str, RiskRule] = {}


def register_rule(name: str) -> Callable[[RiskRule], RiskRule]:
    def decorator(fn: RiskRule) -> RiskRule:
        RISK_RULES[name] = fn
        return fn
    return decorator


@register_rule("max_spread")
def _max_spread(ctx: Dict, threshold: float) -> bool:
    return float(ctx.get("spread_bp", 0.0)) > float(threshold)


@register_rule("gap_guard")
def _gap_guard(ctx: Dict, threshold: float) -> bool:
    return float(ctx.get("gap", 0.0)) > float(threshold)


@register_rule("loss_streak")
def _loss_streak(ctx: Dict, threshold: float) -> bool:
    return int(ctx.get("loss_streak", 0)) >= int(threshold)


@register_rule("illiquidity")
def _illiquidity(ctx: Dict, threshold: float) -> bool:
    return float(ctx.get("depth", float("inf"))) < float(threshold)


@register_rule("max_position")
def _max_position(ctx: Dict, threshold: float) -> bool:
    return abs(float(ctx.get("position", 0.0))) > float(threshold)


@register_rule("drawdown_circuit")
def _drawdown(ctx: Dict, threshold: float) -> bool:
    return float(ctx.get("drawdown", 0.0)) > float(threshold)


__all__ = ["RiskEvent", "RiskManager", "RISK_RULES", "register_rule"]
