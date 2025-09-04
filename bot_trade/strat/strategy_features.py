from __future__ import annotations

"""Strategy feature registry stub.

Plugins can register additional feature builders by updating
``FEATURE_REGISTRY`` at import time. Training code will later select a
builder via :func:`get_feature_builder`.
"""

from typing import Any, Callable, Dict
import numpy as np
import pandas as pd
from bot_trade.data.collectors.base import CollectorConfig, MarketCollector
import warnings
from pathlib import Path
from bot_trade.data.router import DataRouter
from bot_trade.ai_core import pipeline
from bot_trade.ai_core.pipeline import enforce_ai_core_marker

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is a soft dependency
    yaml = None


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": hist})


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()


def _realized_vol(df: pd.DataFrame, window: int) -> pd.Series:
    ret = df["close"].pct_change()
    return ret.rolling(window).std() * (window ** 0.5)


def _rolling(df: pd.Series, window: int) -> pd.Series:
    return df.rolling(window).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def _apply_post(df: pd.DataFrame, post_cfg: dict) -> pd.DataFrame:
    for item in post_cfg or []:
        name, params = next(iter(item.items()))
        if name == "zscore":
            on = params.get("on")
            win = int(params.get("window", 20))
            if on in df:
                df[f"zscore_{on}"] = _zscore(df[on], win)
            else:
                warnings.warn(f"[SIG] zscore source missing: {on}")
    return df


def build_features(df_like, cfg) -> Dict[str, Any]:
    """Build feature dataframe based on signals spec.

    ``cfg`` may provide ``signals_spec`` pointing to a YAML file. The YAML
    describes a ``pipeline`` list where each item is a mapping containing the
    indicator name and its parameters. Missing indicators emit a warning and
    are skipped.
    """
    df = pd.DataFrame(df_like).copy()
    spec_path = cfg.get("signals_spec") if isinstance(cfg, dict) else None
    if not spec_path:
        return {}
    if yaml is None:
        warnings.warn("[SIG] PyYAML not installed; skipping signals")
        return {}
    try:
        spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - config errors
        warnings.warn(f"[SIG] failed to read spec: {exc}")
        return {}
    pipe = spec.get("pipeline", [])
    for item in pipe:
        name, params = next(iter(item.items()))
        try:
            if name == "macd":
                params = {k: int(v) for k, v in params.items()}
                df = df.join(_macd(df["close"], params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)))
            elif name == "rsi":
                period = int(params.get("period", 14))
                df["rsi"] = _rsi(df["close"], period)
            elif name == "atr":
                period = int(params.get("period", 14))
                df["atr"] = _atr(df, period)
            elif name == "realized_vol":
                window = int(params.get("window", 60))
                df["realized_vol"] = _realized_vol(df, window)
            elif name == "spread_bp":
                window = int(params.get("window", 10))
                df["spread_bp_roll"] = _rolling(df["spread_bp"], window)
            elif name == "depth_top":
                window = int(params.get("window", 5))
                df["depth_top_roll"] = _rolling(df["depth_top"], window)
        except Exception as exc:
            warnings.warn(f"[SIG] failed {name}: {exc}")
    df = _apply_post(df, spec.get("post"))
    df.ffill(limit=5, inplace=True)
    df.dropna(how="any", inplace=True)
    return df


def load_via_router(args) -> pd.DataFrame:
    """Route data loading and enforce ai_core pipeline."""

    raw_dir = getattr(args, "raw_dir", None)
    if not raw_dir and getattr(args, "data_dir", None):
        raw_dir = getattr(args, "data_dir")
        print("[DATA] --data-dir is deprecated; using as --raw-dir")

    router = DataRouter(
        mode=getattr(args, "data_mode", "raw"),
        source=getattr(args, "data_source", "csvparquet"),
        raw_dir=raw_dir or "data/ready",
        exchange=getattr(args, "exchange", None),
        cache_dir=getattr(args, "cache_dir", "data/cache"),
    )
    df = router.load(
        symbol=getattr(args, "symbol"),
        frame=getattr(args, "frame"),
        start=getattr(args, "start", None),
        end=getattr(args, "end", None),
        ccxt_symbol=getattr(args, "ccxt_symbol", None),
    )
    features, _meta = pipeline.apply(df, getattr(args, "signals_spec", None))
    assert pipeline.was_applied(), "ai_core pipeline bypass detected"
    paths = getattr(args, "paths", None)
    if paths is not None and hasattr(paths, "performance_dir"):
        enforce_ai_core_marker(paths.performance_dir)
    return features

FEATURE_REGISTRY: Dict[str, Callable[[Any, Dict[str, Any]], Any]] = {
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
