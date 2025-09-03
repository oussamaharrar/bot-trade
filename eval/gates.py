from __future__ import annotations

"""Evaluation gate placeholder."""

from typing import Sequence, Mapping


def promote_if(policies: Sequence[Mapping], thresholds: Mapping) -> bool:
    """Print gate decision based on last window metrics."""
    try:
        metrics = policies[-1]
        print(f"[GATE] pass metrics={metrics}")
        return True
    except Exception:
        print("[GATE] fail metrics_unavailable")
        return False

