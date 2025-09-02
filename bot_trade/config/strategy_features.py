# strategy_features.py (memory-safe, vectorized, configurable)
# - يحسب المؤشرات مرة واحدة بكفاءة (بدون lru_cache على بيانات ضخمة)
# - كل الأعمدة float32 لتقليل الذاكرة
# - تمكين/تعطيل المؤشرات من config.yaml (indicators.enabled)
# - توليد entry_signals بدون حلقات Python (vectorized)
# - خيار تصفية الإشارات mahal عبر filter_signals (افتراضيًا معطل لتفادي البطء على ملفات ضخمة)

import os
import yaml
import logging
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ta
from ta.momentum import AwesomeOscillatorIndicator

from bot_trade.tools.atomic_io import append_jsonl

try:
    CFG_PATH = os.path.join("config", "config.yaml")
except Exception:
    CFG_PATH = "config.yaml"

try:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f) or {}
except FileNotFoundError:
    CONFIG = {}

IND = CONFIG.get("indicators", {})
SIG = CONFIG.get("signals", {})

# نوافذ قابلة للتعديل من الكونفيغ
RSI_WINDOW = IND.get("rsi_window", 14)
BB_WINDOW  = IND.get("bb_window", 20)
BB_DEV     = IND.get("bb_dev", 2)
EMA_WINDOW = IND.get("ema_window", 20)
SMA_WINDOW = IND.get("sma_window", 50)
ATR_WINDOW = IND.get("atr_window", 14)
DC_WINDOW  = IND.get("donchian_window", 20)
ADX_WINDOW = IND.get("adx_window", 14)
VI_WINDOW  = IND.get("vi_window", 14)
KC_WINDOW  = IND.get("kc_window", 20)
KC_ATR_WIN = IND.get("kc_atr_window", 14)
AO_S       = IND.get("ao_short_window", 5)
AO_L       = IND.get("ao_long_window", 34)
STO_WIN    = IND.get("stoch_window", 14)
STO_SMOOTH = IND.get("stoch_smooth_window", 3)

# قائمة تمكين المؤشرات من الكونفيغ: indicators.enabled: ["rsi","bb","macd", ...]
ENABLED_SET = set(map(str, IND.get("enabled", []))) if IND.get("enabled") else None

def on(name: str) -> bool:
    if ENABLED_SET is None:
        return True
    return name in ENABLED_SET


def _as_f32(s: pd.Series) -> pd.Series:
    return s.astype("float32", copy=False)


def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """حساب المؤشرات والإشارات بشكل متجهي وذاكرة-آمن.
    يُحافظ على الأعمدة الموجودة ويضيف أعمدة جديدة فقط.
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()

    # تأمين الأعمدة الأساسية
    for col in ("open","high","low","close","volume"):
        if col not in df.columns:
            df[col] = 0.0
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    if "frame" not in df.columns:
        df["frame"] = "1m"

    # تحويل الأنواع
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float32", copy=False)

    close = df["close"].astype("float32", copy=False)
    high  = df["high"].astype("float32", copy=False)
    low   = df["low"].astype("float32", copy=False)
    vol   = df["volume"].astype("float32", copy=False)

    # RSI
    if on("rsi"):
        df["rsi_14"] = _as_f32(ta.momentum.RSIIndicator(close, window=RSI_WINDOW).rsi())

    # Bollinger Bands
    if on("bb"):
        bb = ta.volatility.BollingerBands(close, window=BB_WINDOW, window_dev=BB_DEV)
        df["bollinger_upper"] = _as_f32(bb.bollinger_hband())
        df["bollinger_lower"] = _as_f32(bb.bollinger_lband())
        df["bollinger_mid"]   = _as_f32(bb.bollinger_mavg())
        # فروقات مع السعر (مفيدة للإشارات/المكافآت)
        df["bollinger_upper_diff"] = (df["bollinger_upper"] - close).astype("float32", copy=False)
        df["bollinger_lower_diff"] = (close - df["bollinger_lower"]).astype("float32", copy=False)

    if on("macd"):
        macd = ta.trend.MACD(close)
        df["macd"] = _as_f32(macd.macd())
        df["macd_signal"] = _as_f32(macd.macd_signal())
        df["macd_hist"] = _as_f32(macd.macd_diff())

    if on("ema"):
        df["ema20"] = _as_f32(ta.trend.EMAIndicator(close, window=EMA_WINDOW).ema_indicator())

    if on("sma"):
        df["sma50"] = _as_f32(ta.trend.SMAIndicator(close, window=SMA_WINDOW).sma_indicator())

    if on("atr"):
        df["atr"] = _as_f32(ta.volatility.AverageTrueRange(high, low, close, window=ATR_WINDOW).average_true_range())

    if on("obv"):
        df["obv"] = _as_f32(ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume())

    if on("cci"):
        df["cci"] = _as_f32(ta.trend.CCIIndicator(high, low, close, window=ATR_WINDOW).cci())

    if on("stoch"):
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=STO_WIN, smooth_window=STO_SMOOTH)
        df["stoch_k"] = _as_f32(stoch.stoch())
        df["stoch_d"] = _as_f32(stoch.stoch_signal())

    if on("adx"):
        df["adx"] = _as_f32(ta.trend.ADXIndicator(high, low, close, window=ADX_WINDOW).adx())

    if on("donchian"):
        hi = ta.volatility.DonchianChannel(high, low, close, window=DC_WINDOW)
        df["donchian_high"] = _as_f32(hi.donchian_channel_hband())
        df["donchian_low"]  = _as_f32(hi.donchian_channel_lband())

    if on("vi"):
        vi = ta.trend.VortexIndicator(high, low, close, window=VI_WINDOW)
        df["vi_pos"] = _as_f32(vi.vortex_indicator_pos())
        df["vi_neg"] = _as_f32(vi.vortex_indicator_neg())

    if on("kc"):
        kc = ta.volatility.KeltnerChannel(high, low, close, window=KC_WINDOW, original_version=True)
        df["kc_upper"] = _as_f32(kc.keltner_channel_hband())
        df["kc_lower"] = _as_f32(kc.keltner_channel_lband())

    if on("ao"):
        ao = AwesomeOscillatorIndicator(high, low, window1=AO_S, window2=AO_L)
        df["ao"] = _as_f32(ao.ao())

    # ===============================
    # Entry signals (vectorized)
    # ===============================
    try:
        bl = df["bollinger_lower"] if "bollinger_lower" in df.columns else (close * 0 - 1)
        bh = df["donchian_high"]  if "donchian_high"  in df.columns else (close * 0 + np.inf)
        dl = df["donchian_low"]   if "donchian_low"   in df.columns else (close * 0 - np.inf)
        adx = df["adx"] if "adx" in df.columns else (close * 0)
        rsi = df["rsi_14"] if "rsi_14" in df.columns else (close * 0 + 50)

        mean_rev = (close < bl) & (rsi < 30)
        trend_f  = (df.get("ema20", close) > df.get("sma50", close))
        brk_buy  = (close > bh) & (adx > 25)
        brk_sell = (close < dl) & (adx > 25)

        df["signal_mean_reversion"] = mean_rev.astype("int8", copy=False)
        df["signal_trend_follow"]   = trend_f.astype("int8", copy=False)
        df["signal_breakout_buy"]   = brk_buy.astype("int8", copy=False)
        df["signal_breakout_sell"]  = brk_sell.astype("int8", copy=False)

        # bitmask مدمج كبديل خفيف للأحرف
        df["entry_code"] = (
            df["signal_mean_reversion"].astype("uint8")
            | (df["signal_trend_follow"].astype("uint8") << 1)
            | (df["signal_breakout_buy"].astype("uint8") << 2)
            | (df["signal_breakout_sell"].astype("uint8") << 3)
        ).astype("uint8", copy=False)

        # بناء نص الإشارات بدون حلقات Python
        parts = (
            np.where(mean_rev.values, "rsi_recovery;", "")
            + np.where(trend_f.values, "macd_bullish;", "")
            + np.where(brk_buy.values, "breakout_buy;", "")
            + np.where(brk_sell.values, "breakout_sell;", "")
        )
        df["entry_signals"] = pd.Series(parts, index=df.index)
    except Exception as e:
        logging.warning("[signals] failed to build entry_signals: %s", e)

    # التعامل مع فئات/نصوص
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if str(df[c].dtype) == 'category':
            if '' not in df[c].cat.categories:
                df[c] = df[c].cat.add_categories([''])
            df[c] = df[c].fillna('').astype('category')
        else:
            df[c] = df[c].fillna('')

    # إجبار float32 في النهاية
    f64 = df.select_dtypes(include=['float64']).columns
    if len(f64):
        df[f64] = df[f64].astype('float32', copy=False)


    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Strategy failure policy (merged from strategy_failure.py)
# ----------------------------------------------------------------------

SF_CONFIG: Dict[str, Any] = {}
SF_STATE: Dict[str, int] = {"level": 0, "cool": 0}


def configure(cfg: Dict[str, Any]) -> None:
    """Set global configuration for strategy failure policy."""
    SF_CONFIG.clear()
    SF_CONFIG.update(cfg or {})


def _now() -> str:
    return dt.datetime.utcnow().isoformat()


def evaluate_step(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return zero or more failure events detected this step."""
    if not SF_CONFIG.get("enabled", False):
        return []
    thr = SF_CONFIG.get("thresholds", {}) or {}
    events: List[Dict[str, Any]] = []
    ts = ctx.get("ts") or _now()
    for key, limit in thr.items():
        val = ctx.get(key)
        if val is None or limit is None:
            continue
        try:
            v = float(val)
            lim = float(limit)
        except Exception:
            continue
        trig = v >= lim if key in {"loss_streak", "stuck_position_s", "partial_fill_timeout_s"} else v > lim
        if trig:
            events.append({
                "flag": key,
                "reason": f"{key} threshold",
                "value": v,
                "threshold": lim,
                "ts": ts,
            })
    return events


def _clamp(key: str, value: float) -> float:
    clamp_cfg = (SF_CONFIG.get("clamps") or {}).get(key, {})
    lo = float(clamp_cfg.get("min", float("-inf")))
    hi = float(clamp_cfg.get("max", float("inf")))
    return max(lo, min(value, hi))


def apply_actions(
    events: List[Dict[str, Any]],
    *,
    risk_manager: Any = None,
    controller: Any = None,
    env: Any = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Escalate actions based on policy and record events."""
    summary: Dict[str, Any] = {"applied_actions": [], "new_risk_bounds": {}}
    if not events:
        if SF_STATE.get("cool", 0) > 0:
            SF_STATE["cool"] -= 1
            if SF_STATE["cool"] <= 0:
                SF_STATE["level"] = 0
        return summary

    SF_STATE["cool"] = int(SF_CONFIG.get("cool_down_steps", 0))
    actions = SF_CONFIG.get("actions", [])
    level = SF_STATE.get("level", 0)

    for ev in events:
        try:
            if risk_manager is not None:
                risk_manager.record_flag(
                    "strategy_failure", ev["flag"], ev["value"], ev["threshold"]
                )
        except Exception:
            pass

    applied: List[str] = []
    new_bounds: Dict[str, float] = {}
    if level < len(actions):
        act = actions[level]
        if act == "reduce_risk" and risk_manager is not None:
            try:
                factor = 0.5
                if getattr(controller, "last_regime", None) in {"high_vol", "low_liquidity"}:
                    factor *= 0.8
                risk_manager.current_risk = _clamp("risk_scale", risk_manager.current_risk * factor)
                new_bounds["risk_scale"] = risk_manager.current_risk
                if env is not None and hasattr(env, "exec_sim"):
                    env.exec_sim.max_spread_bp = _clamp(
                        "max_spread_bp", getattr(env.exec_sim, "max_spread_bp", 0.0) * factor
                    )
                    new_bounds["max_spread_bp"] = env.exec_sim.max_spread_bp
            except Exception:
                pass
        elif act == "freeze_trading" and env is not None:
            try:
                setattr(env, "trading_frozen", True)
            except Exception:
                pass
        elif act == "flat_all" and env is not None:
            try:
                flat = getattr(env, "flat_all", None)
                if callable(flat):
                    flat()
            except Exception:
                pass
        elif act == "halt_training":
            summary["halt"] = True
        applied.append(act)
        level += 1
        if controller is not None and getattr(controller, "log_path", None) and act != "warn":
            record = {"ts": _now(), "source": "safety", "action": act, "risk_bounds": new_bounds}
            try:
                append_jsonl(controller.log_path, record)
            except Exception:
                pass

    SF_STATE["level"] = level

    if log_path is not None:
        for ev in events:
            rec = dict(ev)
            rec["actions"] = applied
            try:
                append_jsonl(log_path, rec)
            except Exception:
                pass

    summary["applied_actions"] = applied
    summary["new_risk_bounds"] = new_bounds
    return summary

