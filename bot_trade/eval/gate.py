from __future__ import annotations

"""Evaluation gate helper loading thresholds from YAML."""

from pathlib import Path
from typing import Mapping, Tuple, List

import yaml


_DEFAULT_PATH = Path("config") / "eval" / "gate.yml"


def load_thresholds(path: Path | None = None) -> Mapping[str, Mapping[str, float]]:
    p = path or _DEFAULT_PATH
    try:
        with Path(p).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out[str(k)] = {kk: float(vv) for kk, vv in v.items() if kk in {"min", "max"}}
    return out


def check(metrics: Mapping[str, float | None], thresholds: Mapping[str, Mapping[str, float]]) -> Tuple[bool, List[str]]:
    """Return (pass, reasons) for metrics against thresholds."""

    reasons: List[str] = []
    passed = True
    for key, cond in thresholds.items():
        val = metrics.get(key)
        if val is None:
            passed = False
            reasons.append(f"{key}_missing")
            continue
        min_v = cond.get("min")
        max_v = cond.get("max")
        if min_v is not None and val < min_v:
            passed = False
            reasons.append(f"{key}<{min_v}")
        if max_v is not None and val > max_v:
            passed = False
            reasons.append(f"{key}>{max_v}")
    return passed, reasons


def gate_metrics(metrics: Mapping[str, float | None], path: Path | None = None) -> Tuple[bool, List[str]]:
    thr = load_thresholds(path)
    return check(metrics, thr)

