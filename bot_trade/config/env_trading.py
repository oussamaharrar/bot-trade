# env_trading.py – Gymnasium-compatible trading environment (patched)
from __future__ import annotations
import os, json, time, logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from gymnasium import Env, spaces

from bot_trade.config.risk_manager import RiskManager
from bot_trade.signals.entry_signals import generate_entry_signals
from bot_trade.signals.danger_signals import detect_danger_signals
from bot_trade.signals.freeze_signals import check_freeze_conditions
from bot_trade.signals.recovery_signals import compute_recovery_signals
from bot_trade.config.strategy_features import add_strategy_features
from bot_trade.signals.reward_signals import RewardSignalTracker
from bot_trade.ai_core.entry_verifier import smart_entry_guard

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

class TradingEnv(Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 60}

    def __init__(self, data: pd.DataFrame, frame: Optional[str] = None, symbol: Optional[str] = None,
                 initial_balance: float = 1000.0, stop_loss: float = 0.02, take_profit: float = 0.04,
                 max_loss_per_session: float = 0.2, max_consecutive_losses: int = 10, max_steps: int = 3_000,
                 use_indicators: bool = True, price_col: str = "close", writers=None, **kwargs):
        super().__init__()
        self.config = kwargs.get("config") or {}
        self.frame = frame or str(self.config.get("DEFAULT_FRAME", "1m"))
        self.symbol = (symbol or self.config.get("DEFAULT_SYMBOL", "BTCUSDT")).upper()
        self.session_id = self.config.get("session_id")
        # writers passed explicitly
        self.writers = writers if writers is not None else (kwargs.get("writers") if isinstance(kwargs, dict) else None)
        self.safe = bool(self.config.get("safe", False))
        self.price_col = price_col
        self.use_indicators = use_indicators

        self.ptr = 0; self.steps = 0
        self.usdt = float(initial_balance); self.coin = 0.0
        self.entry_price = None; self.risk_pct = 1.0
        self.win_streak = 0; self.loss_streak = 0
        self.prev_value = self.usdt
        self.equity_curve: list[float] = []; self.max_drawdown = 0.0

        # ========== Data ==========
        if data is None or len(data) == 0:
            raise ValueError("TradingEnv requires non-empty DataFrame")

        df = data.copy()
        if self.use_indicators:
            df = add_strategy_features(df)

        if price_col not in df.columns:
            raise ValueError(f"Price column '{price_col}' is missing in DataFrame")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns available to form observations/price.")

        df.drop(columns=["frame"], errors="ignore", inplace=True)
        if "datetime" in df.columns:
            self.df = df.set_index(["symbol","datetime"])
        else:
            # fallback: استخدم الفهرس الحالي كمستوى زمني إن لم تتوفر عمود datetime
            self.df = df.set_index(["symbol", df.index])
        self.symbols = tuple(sorted(self.df.index.get_level_values(0).unique()))
        self.current_symbol = self.symbol if self.symbol in self.symbols else self.symbols[0]

        # runtime
        self.initial_balance = float(initial_balance)
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_loss_per_session = max_loss_per_session
        self.max_consecutive_losses = max_consecutive_losses
        self.max_steps = int(max_steps)

        rm_cfg = (self.config or {}).get("risk_manager", {})
        risk_log = None
        try:
            if writers is not None and hasattr(writers, "paths"):
                risk_log = writers.paths.get("risk_csv")
        except Exception:
            pass
        self.risk_engine = RiskManager(
            dynamic_sizing=rm_cfg.get("dynamic_sizing", True),
            min_pct=rm_cfg.get("min_pct", 0.1),
            max_pct=rm_cfg.get("max_pct", 1.0),
            max_drawdown_stop=rm_cfg.get("max_drawdown_stop", 0.3),
            log_path=risk_log,
        )
        rw_cfg = (self.config or {}).get("reward_shaping", {})
        self.reward_tracker = RewardSignalTracker(rw_cfg)
        self.current_signals: Dict[str, int] = {}

        # spaces
        self.obs_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "entry_signals" in self.obs_cols: self.obs_cols.remove("entry_signals")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32)

    # -------- Helpers --------
    @property
    def single_action_space(self):
        return self.action_space

    @property
    def single_observation_space(self):
        return self.observation_space

    def _price(self) -> float:
        try:
            return float(self._row()[self.price_col])
        except Exception:
            return float("nan")

    def _row(self) -> pd.Series:
        try:
            return self.df.loc[self.current_symbol].iloc[self.ptr]
        except Exception:
            return self.df.iloc[self.ptr]

    def _symbol_len(self) -> int:
        try:
            return len(self.df.loc[self.current_symbol])
        except Exception:
            return len(self.df)

    def _make_obs(self):
        try:
            row = self._row()
            obs = row[self.obs_cols].astype(float).values.astype(np.float32)
            return obs
        except Exception:
            return np.zeros((len(self.obs_cols),), dtype=np.float32)

    def _normalize_entry_signals(self, raw_entry: Any, row: pd.Series) -> Dict[str, int]:
        entry_sigs: Dict[str, int] = {}
        if isinstance(raw_entry, dict):
            entry_sigs = {str(k): int(1 if bool(v) and float(v) != 0 else 0) for k,v in raw_entry.items()}
        elif isinstance(raw_entry, (list, tuple, set)):
            entry_sigs = {str(k): 1 for k in raw_entry}
        elif isinstance(raw_entry, str):
            tokens = [t.strip() for t in raw_entry.split(";") if t and t.strip()]
            entry_sigs = {t: 1 for t in tokens}
        else:
            entry_sigs = {}
        for k in ("signal_trend_follow","signal_mean_reversion","signal_breakout_buy","signal_breakout_sell"):
            if k in row.index:
                entry_sigs[k] = int(entry_sigs.get(k,0) or int(row.get(k,0)))
        return entry_sigs

    def _build_df_slice(self) -> pd.DataFrame:
        try:
            return self.df.loc[self.current_symbol][[self.price_col]].reset_index(drop=True)
        except Exception:
            try:
                numeric = self.df.loc[self.current_symbol].select_dtypes(include=[np.number])
                if numeric.shape[1] == 0:
                    return pd.DataFrame({self.price_col: [self._price()]})
                return numeric[[self.price_col]].reset_index(drop=True)
            except Exception:
                return pd.DataFrame({self.price_col: [self._price()]})

    def _log_decision(self, payload: Dict[str, Any]) -> None:
        try:
            if self.writers is not None and hasattr(self.writers, "log_decision"):
                self.writers.log_decision(**payload)
        except Exception:
            pass

    # -------- Gym API --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.ptr = 0; self.steps = 0
        self.usdt = float(self.initial_balance); self.coin = 0.0
        self.entry_price = None; self.risk_pct = 1.0
        self.win_streak = 0; self.loss_streak = 0
        self.prev_value = self.initial_balance
        self.equity_curve.clear(); self.max_drawdown = 0.0
        obs = self._make_obs()
        info = {"symbol": self.current_symbol, "frame": self.frame}
        return obs, info

    def step(self, action: int):
        price = self._price(); row = self._row()

        vol = float(row.get("atr", row.get("volatility", 0.0)))
        danger = bool(detect_danger_signals(row=row, volatility=vol))
        freeze = bool(check_freeze_conditions(row=row, volatility=vol))
        recovery = int(compute_recovery_signals(row=row, volatility=vol))

        raw_entry = generate_entry_signals(row)
        entry_sigs = self._normalize_entry_signals(raw_entry, row)
        self.current_signals = entry_sigs

        risk_out = self.risk_engine.update(
            reward=0.0,
            indicators={
                "rsi_14": float(row.get("rsi_14", 50.0)),
                "macd": float(row.get("macd", 0.0)),
                "macd_signal": float(row.get("macd_signal", 0.0)),
                "adx": float(row.get("adx", 20.0)),
            },
            drawdown=float(self.max_drawdown),
            volatility=vol,
            recovery_signals=recovery,
            signals={
                "signal_trend_follow": int(entry_sigs.get("signal_trend_follow", 0)),
                "signal_mean_reversion": int(entry_sigs.get("signal_mean_reversion", 0)),
                "signal_breakout_buy": int(entry_sigs.get("signal_breakout_buy", 0)),
                "signal_breakout_sell": int(entry_sigs.get("signal_breakout_sell", 0)),
                "feature_summary": int(sum(int(v) for v in entry_sigs.values() if isinstance(v, (int, float)))),
            },
        )
        try:
            self.risk_pct = float(np.clip(float(risk_out.get("risk_pct", self.risk_pct)), 0.0, 1.0))
        except Exception:
            pass

        decision_reason = "none"; allowed = True
        traded_qty = 0.0; trade_side = "HOLD"

        if freeze:
            action = 0
            decision_reason = "freeze"

        ai_ctrl = (self.config.get("ai_control") or {})
        if action in (1, 2):
            active_list = [k for k, v in entry_sigs.items() if int(v) == 1]
            try:
                df_slice = self._build_df_slice()
                allowed = smart_entry_guard(self.frame, self.current_symbol, df_slice=df_slice,
                                            signal_idx=self.ptr, active_signals=active_list, session_id=self.session_id)
            except TypeError:
                try:
                    from bot_trade.ai_core.entry_verifier import verify_entry
                    allowed = verify_entry(self.frame, self.current_symbol, active_list, self._build_df_slice(),
                                           self.ptr, threshold=ai_ctrl.get("simulation_threshold", 0.5), session_id=self.session_id)
                except Exception:
                    allowed = True
            if not allowed:
                action = 0
                decision_reason = "ai_guard"

        if danger:
            self.risk_pct = float(np.clip(self.risk_pct * 0.5, 0.0, 1.0))

        fee = float(self.config.get("fees", {}).get("commission_pct", 0.0))
        if action == 1 and self.usdt > 0.0:
            buy_amt = self.usdt * float(np.clip(self.risk_pct, 0.0, 1.0))
            qty = buy_amt / price
            cost = buy_amt * (1.0 + fee)
            if cost <= self.usdt and qty > 0.0:
                self.usdt -= cost
                self.coin += qty
                if self.entry_price is None:
                    self.entry_price = price
                traded_qty = qty
                trade_side = "BUY"
        elif action == 2 and self.coin > 1e-12:
            sell_qty = self.coin * float(np.clip(self.risk_pct, 0.0, 1.0))
            proceeds = sell_qty * price * (1.0 - fee)
            self.usdt += proceeds
            self.coin -= sell_qty
            traded_qty = sell_qty
            trade_side = "SELL"

        value = self.usdt + self.coin * price
        pnl = value - self.prev_value
        self.prev_value = value

        self.equity_curve.append(value)
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            dd = (peak - value) / max(peak, 1e-9)
            self.max_drawdown = max(self.max_drawdown, dd)
        pnl_pct = pnl / max(self.initial_balance, 1e-9)
        trade_cost = abs(traded_qty) * price * fee / max(self.initial_balance, 1e-9)
        dwell_pen = 1.0 if self.coin > 0 and action == 0 else 0.0
        reward, comps = self.reward_tracker.step_reward(
            pnl_pct,
            self.equity_curve,
            atr=row.get("atr"),
            drawdown=self.max_drawdown,
            danger=danger,
            freeze=freeze,
            trade_cost=trade_cost,
            dwell_pen=dwell_pen,
            signal_trend_follow=bool(entry_sigs.get("signal_trend_follow", 0)),
        )

        terminated = False; term_reason = "none"
        if value <= (1.0 - self.max_loss_per_session) * self.initial_balance:
            terminated = True; term_reason = "max_session_loss"
        if self.loss_streak >= self.max_consecutive_losses:
            terminated = True; term_reason = "max_consecutive_losses"

        if pnl < 0: self.loss_streak += 1; self.win_streak = 0
        elif pnl > 0: self.win_streak += 1; self.loss_streak = 0

        info = {
            "symbol": self.current_symbol, "frame": self.frame, "price": float(price),
            "usdt": float(self.usdt), "coin": float(self.coin), "value": float(value),
            "pnl": float(pnl), "reward": float(reward), "drawdown": float(self.max_drawdown),
            "risk_pct": float(self.risk_pct), "freeze": bool(freeze), "danger": bool(danger),
            "recovery": int(recovery), "win_streak": int(self.win_streak), "loss_streak": int(self.loss_streak),
            "signals_active": [k for k,v in entry_sigs.items() if int(v)==1],
            "entry_blocked": (decision_reason == "ai_guard"), "decision_reason": decision_reason,
            "term_reason": term_reason,
        }
        comps["pnl"] = float(pnl_pct)
        comps["trade_cost"] = float(trade_cost)
        comps["dwell_pen"] = float(dwell_pen)
        info["reward_components"] = comps
        if trade_side in ("BUY","SELL") and traded_qty > 0:
            info["trade"] = {"side": trade_side, "price": float(price), "size": float(traded_qty),
                             "pnl": float(pnl), "equity": float(value), "reason": decision_reason}

        self._log_decision({
            "step": int(self.steps), "ptr": int(self.ptr), "action": int(action),
            "trade_side": trade_side, "allowed": bool(allowed), "decision_reason": decision_reason,
            "entry_blocked": (decision_reason == "ai_guard"), "risk_pct": float(self.risk_pct),
            "price": float(price), "traded_qty": float(traded_qty), "pnl": float(pnl), "value": float(value),
            "danger": bool(danger), "freeze": bool(freeze), "active_signals": [k for k,v in entry_sigs.items() if int(v)==1],
            "term_reason": term_reason,
        })

        self.steps += 1; self.ptr += 1
        truncated = self.steps >= self.max_steps or self.ptr >= self._symbol_len()

        obs = self._make_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info
