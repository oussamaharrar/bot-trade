from __future__ import annotations

"""Strategy feature registry stub.

Plugins can register additional feature builders by updating
``FEATURE_REGISTRY`` at import time. Training code will later select a
builder via :func:`get_feature_builder`.
"""

from typing import Any, Callable, Dict
import numpy as np


def build_features(df_like, cfg) -> Dict[str, Any]:
    """Baseline feature builder returning an empty feature set."""
    return {}

FEATURE_REGISTRY: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = {
    "baseline": build_features,
}


def get_feature_builder(name: str) -> Callable[[Any, Dict[str, Any]], Dict[str, Any]]:
    """Select a feature builder by name from ``FEATURE_REGISTRY``."""
    return FEATURE_REGISTRY.get(name, build_features)


_INJECTED: set[str] = set()


def read_exogenous_signals(run_paths, max_rows: int = 2048) -> dict[str, np.ndarray]:
    """Safe reader for signals.jsonl → returns dict of exogenous feature arrays."""
    import json, math
    import numpy as np
    from bot_trade.config.rl_paths import memory_dir

    run_id = getattr(run_paths, "run_id", None) or (
        run_paths.get("run_id") if isinstance(run_paths, dict) else None
    )
    path = memory_dir() / "Knowlogy" / "signals.jsonl"
    if not path.exists():
        return {}
    data: dict[str, list[float]] = {}
    confidences: list[float] = []
    sources: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx >= max_rows:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            try:
                val = float(obj.get("value"))
            except Exception:
                continue
            if not math.isfinite(val):
                continue
            sig = obj.get("signal")
            data.setdefault(sig, []).append(val)
            conf = obj.get("confidence")
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = float("nan")
            if math.isfinite(conf_f):
                confidences.append(conf_f)
            prov = obj.get("provenance", {})
            src = prov.get("collector")
            if src:
                sources.add(src)
    out = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    count = sum(len(v) for v in data.values())
    key = str(run_id)
    if count and key not in _INJECTED:
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        print(
            f"[AI_CORE] signals injected count={count} sources={sorted(sources)} confidence_mean={mean_conf:.2f}"
        )
        _INJECTED.add(key)
    return out
