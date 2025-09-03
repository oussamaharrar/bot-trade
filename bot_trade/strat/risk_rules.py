from __future__ import annotations
"""Risk management helpers with circuit breakers and kill switch."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
