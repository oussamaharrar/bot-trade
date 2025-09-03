"""Evaluation gate applying threshold checks loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple, List

import yaml

_DEFAULT_CFG = Path("config") / "eval_gate.yaml"


def _load(path: Path | None = None) -> Mapping[str, float | bool]:
    p = path or _DEFAULT_CFG
    try:
        with Path(p).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    return {str(k): v for k, v in data.items()}


def gate_metrics(metrics: Mapping[str, float | None], wfa_pass: bool = True, path: Path | None = None) -> Tuple[bool, List[str]]:
    cfg = _load(path)
    reasons: List[str] = []
    def _check(key: str, cmp, thr):
        val = metrics.get(key)
        if val is None or not cmp(val, thr):
            reasons.append(f"{key}")
    if (thr := cfg.get("min_sharpe")) is not None:
        _check("sharpe", lambda v, t: v >= t, float(thr))
    if (thr := cfg.get("max_drawdown")) is not None:
        _check("max_drawdown", lambda v, t: v <= t, float(thr))
    if (thr := cfg.get("min_win_rate")) is not None:
        _check("win_rate", lambda v, t: v >= t, float(thr))
    if (thr := cfg.get("max_turnover")) is not None:
        _check("turnover", lambda v, t: v <= t, float(thr))
    if cfg.get("require_walk_forward_pass") and not wfa_pass:
        reasons.append("walk_forward")
    return (len(reasons) == 0, reasons)


__all__ = ["gate_metrics"]
