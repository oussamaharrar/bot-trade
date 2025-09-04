"""Threshold based evaluation gates."""

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class GateResult:
    passed: bool
    reasons: list[str]
    pass_ratio: float
    promote_if: bool = False

    def to_dict(self) -> dict[str, object]:  # pragma: no cover - trivial
        return asdict(self)


def threshold_gate(
    metrics: Mapping[str, float | None],
    thresholds: Mapping[str, float],
    promote_if: bool = False,
    promotion_path: Path | None = None,
) -> GateResult:
    """Evaluate ``metrics`` against ``thresholds``.

    Threshold keys:
        - ``min_sharpe``
        - ``min_sortino``
        - ``max_drawdown``
        - ``min_winrate`` (maps to ``win_rate`` metric)

    When ``promote_if`` is ``True`` and all checks pass, a small JSON record is
    written to ``promotion_path`` if provided.
    """

    reasons: list[str] = []
    checks = 0
    passes = 0

    def _check(key: str, metric_key: str, cmp, thr):
        nonlocal checks, passes
        checks += 1
        val = metrics.get(metric_key)
        if val is None or not cmp(val, thr):
            reasons.append(key)
        else:
            passes += 1

    if "min_sharpe" in thresholds:
        _check("min_sharpe", "sharpe", lambda v, t: v >= t, float(thresholds["min_sharpe"]))
    if "min_sortino" in thresholds:
        _check("min_sortino", "sortino", lambda v, t: v >= t, float(thresholds["min_sortino"]))
    if "max_drawdown" in thresholds:
        _check("max_drawdown", "max_drawdown", lambda v, t: v <= t, float(thresholds["max_drawdown"]))
    if "min_winrate" in thresholds:
        _check("min_winrate", "win_rate", lambda v, t: v >= t, float(thresholds["min_winrate"]))

    passed = not reasons
    pass_ratio = passes / checks if checks else 1.0

    result = GateResult(passed=passed, reasons=reasons, pass_ratio=pass_ratio, promote_if=promote_if and passed)

    if result.promote_if and promotion_path is not None:
        record = {"ts": time.time(), "metrics": dict(metrics)}
        try:
            promotion_path.parent.mkdir(parents=True, exist_ok=True)
            with promotion_path.open("w", encoding="utf-8") as fh:
                json.dump(record, fh)
        except OSError:
            pass  # best effort

    return result


__all__ = ["GateResult", "threshold_gate"]
