from __future__ import annotations

"""Minimal ai_core feature pipeline."""

from pathlib import Path
import json
from typing import Tuple, Dict, Any

import pandas as pd

_last_apply_called = False


def apply(df: pd.DataFrame, signals_cfg: str | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate and apply signal pipeline to ``df``.

    Returns the transformed dataframe and metadata about applied signals.
    A single JSONL log line is appended to ``ai_core_log.jsonl``.
    """

    global _last_apply_called
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise AssertionError(f"missing column: {col}")
    features = df.copy()
    meta: Dict[str, Any] = {"signals": []}
    if signals_cfg:
        from bot_trade.strat.strategy_features import build_features

        features = build_features(df, {"signals_spec": signals_cfg})
        meta["signals"] = list(features.columns)
    features.dropna(how="any", inplace=True)
    _last_apply_called = True
    log_line = {"rows": int(len(features)), "signals_cfg": signals_cfg or "none"}
    try:
        with (Path("ai_core_log.jsonl")).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_line) + "\n")
    except Exception:
        pass
    return features, meta


def was_applied() -> bool:
    return _last_apply_called

