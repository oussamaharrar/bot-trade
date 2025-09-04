from __future__ import annotations

"""Path helpers for algorithm/scoped directories."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunPaths:
    root: Path

    @property
    def agents(self) -> Path:
        p = self.root / "agents"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results(self) -> Path:
        p = self.root / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p


def algo_root(base: str, algo: str, symbol: str, frame: str, run_id: str) -> RunPaths:
    root = Path(base) / algo / symbol / frame / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(root)

