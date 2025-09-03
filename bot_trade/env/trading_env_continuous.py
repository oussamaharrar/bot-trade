from __future__ import annotations
import numpy as np
from gymnasium import spaces
from bot_trade.config.env_trading import TradingEnv

class TradingEnvContinuous(TradingEnv):
    """Continuous-action variant of TradingEnv.

    Actions are target positions in [-1,1] where -1 is full short and +1 is
    full long.  Execution uses the existing ExecutionSim and risk manager.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        print("[ENV] action_space=Box(-1.0,1.0) dims=1")

    def step(self, action):
        import numpy as _np
        if getattr(self, "trading_frozen", False):
            target = 0.0
        else:
            target = float(_np.clip(_np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        price = self._price(); row = self._row()
        vol = float(row.get("atr", row.get("volatility", 0.0)))
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
            recovery_signals=0,
            signals={"feature_summary": 0},
        )
        try:
            self.risk_pct = float(_np.clip(float(risk_out.get("risk_pct", self.risk_pct)), 0.0, 1.0))
        except Exception:
            pass
        spread = float(row.get("spread", price * 0.0))
        vol_val = float(row.get("volatility", 0.0))
        depth = row.get("depth")
        spread_bp = (spread / price * 10_000.0) if price > 0 else 0.0
        risk_flag_info = None
        if spread_bp > self.exec_sim.max_spread_bp:
            risk_flag_info = (
                "spread_jump",
                "execution spread limit",
                spread_bp,
                self.exec_sim.max_spread_bp,
            )
            target = 0.0
        else:
            cb = self.risk_engine.check_circuit_breakers(
                spread_bp=spread_bp,
                depth=depth,
            )
            if cb:
                risk_flag_info = cb
                target = 0.0
        if self.risk_engine.cooldown_active():
            target = 0.0
        value = self.usdt + self.coin * price
        target_coin = target * value / max(price, 1e-9)
        orig_target_coin = target_coin
        if self.risk_engine.max_units is not None:
            max_u = self.risk_engine.max_units
            target_coin = float(_np.clip(target_coin, -max_u, max_u))
            if target_coin != orig_target_coin:
                risk_flag_info = ("exposure_cap", "units", target_coin, max_u)
        if self.risk_engine.max_notional is not None:
            max_n = self.risk_engine.max_notional / max(price, 1e-9)
            prev_target = target_coin
            target_coin = float(_np.clip(target_coin, -max_n, max_n))
            if target_coin != prev_target:
                risk_flag_info = (
                    "exposure_cap",
                    "notional",
                    target_coin * price,
                    self.risk_engine.max_notional,
                )
        delta_coin = target_coin - self.coin
        traded_qty = 0.0
        trade_side = None
        exec_res = None
        if abs(delta_coin) > 1e-12:
            side = "buy" if delta_coin > 0 else "sell"
            exec_res = self.exec_sim.execute(side, abs(delta_coin), self.steps, price, spread, vol=vol_val, depth=depth)
            if side == "buy":
                cost = exec_res["filled_qty"] * exec_res["avg_price"] + exec_res["fees"]
                if cost <= self.usdt:
                    self.usdt -= cost
                    self.coin += exec_res["filled_qty"]
                    traded_qty = exec_res["filled_qty"]
                    trade_side = "BUY"
            else:
                proceeds = exec_res["filled_qty"] * exec_res["avg_price"] - exec_res["fees"]
                self.usdt += proceeds
                self.coin -= exec_res["filled_qty"]
                traded_qty = exec_res["filled_qty"]
                trade_side = "SELL"
        value = self.usdt + self.coin * price
        pnl = value - self.prev_value
        self.prev_value = value
        cb_post = self.risk_engine.check_circuit_breakers(pnl=pnl)
        if cb_post and not risk_flag_info:
            risk_flag_info = cb_post
        self.equity_curve.append(value)
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            dd = (peak - value) / max(peak, 1e-9)
            self.max_drawdown = max(self.max_drawdown, dd)
        pnl_pct = pnl / max(self.initial_balance, 1e-9)
        fee = float(self.exec_sim.fee_bp) / 10_000.0
        trade_cost = abs(traded_qty) * price * fee / max(self.initial_balance, 1e-9)
        dwell_pen = 1.0 if abs(self.coin) > 0 and abs(delta_coin) < 1e-12 else 0.0
        reward, comps = self.reward_tracker.step_reward(
            pnl_pct,
            self.equity_curve,
            atr=row.get("atr"),
            drawdown=self.max_drawdown,
            danger=False,
            freeze=False,
            trade_cost=trade_cost,
            dwell_pen=dwell_pen,
            signal_trend_follow=False,
        )
        terminated = False
        term_reason = "none"
        if value <= (1.0 - self.max_loss_per_session) * self.initial_balance:
            terminated = True
            term_reason = "max_session_loss"
        if self.loss_streak >= self.max_consecutive_losses:
            terminated = True
            term_reason = "max_consecutive_losses"
        if pnl < 0:
            self.loss_streak += 1
            self.win_streak = 0
        elif pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        info = {
            "symbol": self.current_symbol,
            "frame": self.frame,
            "price": float(price),
            "usdt": float(self.usdt),
            "coin": float(self.coin),
            "value": float(value),
            "pnl": float(pnl),
            "reward": float(reward),
            "drawdown": float(self.max_drawdown),
            "risk_pct": float(self.risk_pct),
            "win_streak": int(self.win_streak),
            "loss_streak": int(self.loss_streak),
            "term_reason": term_reason,
        }
        if exec_res:
            info.update({
                "trade": {
                    "side": trade_side,
                    "price": float(price),
                    "size": float(traded_qty),
                    "pnl": float(pnl),
                    "equity": float(value),
                }
            })
        if risk_flag_info:
            flag, reason, val, thr = risk_flag_info
            self.risk_engine.record_flag(flag, reason, val, thr)
            info.update({
                "risk_flag": flag,
                "flag_reason": reason,
                "risk_value": val,
                "risk_threshold": thr,
            })
        info["reward_components"] = comps
        self._log_decision({
            "step": int(self.steps),
            "ptr": int(self.ptr),
            "action": float(target),
            "trade_side": trade_side,
            "price": float(price),
            "traded_qty": float(traded_qty),
            "pnl": float(pnl),
            "value": float(value),
            "risk_pct": float(self.risk_pct),
            "term_reason": term_reason,
        })
        self.risk_engine.step_cooldown()
        self.steps += 1
        self.ptr += 1
        truncated = self.steps >= self.max_steps or self.ptr >= self._symbol_len()
        obs = self._make_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info
