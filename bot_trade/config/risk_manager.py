import logging
import os
import time
from typing import Optional, Dict, List
import csv

class RiskManager:
    """Dynamic risk manager using reward EMA, volatility, drawdown, and indicator signals."""

    def __init__(
        self,
        dynamic_sizing: bool = True,
        min_pct: float = 0.1,
        max_pct: float = 1.0,
        max_drawdown_stop: float = 0.3,
        ema_alpha: float = 0.05,
        reward_threshold: float = 0.5,
        volatility_threshold: float = 2.0,
        freeze_limit: int = 3,
        unfreeze_patience: int = 5,
        unfreeze_threshold: float = 0.1,
        log_path: Optional[str] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.dynamic_sizing = bool(dynamic_sizing)
        self.max_risk = float(max_pct)
        self.min_risk = float(min_pct)
        self.ema_alpha = float(ema_alpha)
        self.ema_reward = 0.0
        self.reward_threshold = float(reward_threshold)
        self.volatility_threshold = float(volatility_threshold)
        self.drawdown_limit = float(max_drawdown_stop)
        self.freeze_limit = int(freeze_limit)
        self.unfreeze_patience = int(unfreeze_patience)
        self.unfreeze_threshold = float(unfreeze_threshold)

        self.freeze_mode = False
        self.freeze_counter = 0
        self.current_risk = (self.max_risk + self.min_risk) / 2.0
        self.loss_streak = 0
        self.max_drawdown = 0.0

        self.risk_log_path = log_path
        if self.risk_log_path:
            os.makedirs(os.path.dirname(self.risk_log_path), exist_ok=True)
            with open(self.risk_log_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["reason", "risk_pct", "ema_reward", "drawdown", "freeze_mode"])
        self._last_multi_log = 0.0

    def log_risk_reason(self, reason: str):
        level = logging.INFO
        throttle = 0.0
        if reason == "Multiple entry signals active":
            level = logging.DEBUG
            throttle = 5.0
            now = time.time()
            if now - self._last_multi_log < throttle:
                return
            self._last_multi_log = now
        if self.risk_log_path:
            with open(self.risk_log_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    reason,
                    f"{self.current_risk:.4f}",
                    f"{self.ema_reward:.4f}",
                    f"{self.max_drawdown:.4f}",
                    str(self.freeze_mode)
                ])
        self.logger.log(level, f"[RISK_LOG] {reason} | risk={self.current_risk:.3f}")

    def update(
        self,
        reward: float,
        indicators: Optional[Dict] = None,
        drawdown: float = 0.0,
        volatility: float = 0.0,
        recovery_signals: int = 0,
        signals: Optional[Dict] = None
    ) -> Dict:
        if not self.dynamic_sizing:
            self.ema_reward = (1 - self.ema_alpha) * self.ema_reward + self.ema_alpha * float(reward)
            self.max_drawdown = max(self.max_drawdown, float(drawdown))
            return {
                "risk_pct": self.max_risk,
                "danger_mode": False,
                "freeze_mode": False,
                "notes": [],
                "ema_reward": self.ema_reward,
                "max_drawdown": self.max_drawdown,
            }

        self.ema_reward = (1 - self.ema_alpha) * self.ema_reward + self.ema_alpha * float(reward)
        self.max_drawdown = max(self.max_drawdown, float(drawdown))

        if reward < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        if self.loss_streak >= self.freeze_limit and not self.freeze_mode:
            self.freeze_mode = True
            self.freeze_counter = 0
            self.log_risk_reason("Entering freeze mode due to loss streak")

        if self.freeze_mode:
            if (self.ema_reward > self.unfreeze_threshold) or (int(recovery_signals) >= 2):
                self.freeze_counter += 1
                if self.freeze_counter >= self.unfreeze_patience:
                    self.freeze_mode = False
                    self.freeze_counter = 0
                    self.current_risk = self.min_risk + 0.05
                    self.recovery_counter = 0
                    self.log_risk_reason("Exiting freeze mode after recovery")
            else:
                self.freeze_counter = 0

        if not self.freeze_mode:
            if hasattr(self, 'recovery_counter'):
                self.recovery_counter += 1
                if (self.ema_reward > 0.2) and (self.recovery_counter >= 3):
                    self.current_risk += 0.05
                    self.log_risk_reason("Gradual increase after freeze recovery")
                    if self.current_risk >= self.max_risk:
                        del self.recovery_counter
            else:
                if (self.ema_reward > self.reward_threshold) and (self.max_drawdown < self.drawdown_limit):
                    delta = min(self.max_risk - self.current_risk, self.ema_reward * 0.1)
                    self.current_risk += max(0.0, delta)
                    self.log_risk_reason("Positive reward, low drawdown")
                elif (self.ema_reward < -self.reward_threshold) or (self.max_drawdown >= self.drawdown_limit):
                    delta = min(self.current_risk - self.min_risk, abs(self.ema_reward) * 0.1 + 0.05)
                    self.current_risk -= max(0.0, delta)
                    self.log_risk_reason("Negative reward or high drawdown")

        notes: List[str] = []
        if signals:
            trend = signals.get("signal_trend_follow", 0)
            reversion = signals.get("signal_mean_reversion", 0)
            breakout_sell = signals.get("signal_breakout_sell", 0)
            summary = float(signals.get("feature_summary", 0.0))
            ml = signals.get("ml_signal", 0)

            if trend and indicators and indicators.get("adx", 0) > 25:
                self.current_risk += 0.05
                reason = "Trend-follow signal with strong ADX"
                notes.append(reason)
                self.log_risk_reason(reason)
            if reversion and indicators and indicators.get("adx", 0) > 25:
                self.current_risk -= 0.05
                reason = "Mean-reversion signal penalized (high ADX)"
                notes.append(reason)
                self.log_risk_reason(reason)
            if breakout_sell:
                self.current_risk -= 0.03
                reason = "Breakout sell signal"
                notes.append(reason)
                self.log_risk_reason(reason)
            if ml == 1:
                self.current_risk += 0.03
                reason = "ML signal bullish"
                notes.append(reason)
                self.log_risk_reason(reason)
            if summary >= 2:
                self.current_risk += 0.02
                reason = "Multiple entry signals active"
                notes.append(reason)
                self.log_risk_reason(reason)

        self.current_risk = float(max(self.min_risk, min(self.max_risk, self.current_risk)))

        danger = False
        if float(volatility) > self.volatility_threshold:
            danger = True
            notes.append("High market volatility")
        if self.max_drawdown >= self.drawdown_limit:
            danger = True
            notes.append("Persistent drawdown")
        if indicators and self._detect_indicator_conflict(indicators):
            danger = True
            notes.append("Indicator conflict detected")
        if self.freeze_mode:
            danger = True
            notes.append("Freeze mode active")

        self.logger.debug(
            f"[RiskManager] ema_reward={self.ema_reward:.3f}, drawdown={self.max_drawdown:.3f}, "
            f"risk_pct={self.current_risk:.3f}, freeze={self.freeze_mode}, danger={danger}"
        )

        return {
            "risk_pct": self.current_risk,
            "danger_mode": danger,
            "freeze_mode": self.freeze_mode,
            "notes": notes,
            "ema_reward": self.ema_reward,
            "max_drawdown": self.max_drawdown
        }

    def _detect_indicator_conflict(self, indicators: Dict) -> bool:
        def _to_float(x):
            try:
                if hasattr(x, "item"):
                    x = x.item()
                return float(x)
            except Exception:
                return None

        rsi = _to_float(indicators.get('rsi_14'))
        macd = _to_float(indicators.get('macd'))
        macd_signal = _to_float(indicators.get('macd_signal'))
        if rsi is None or macd is None or macd_signal is None:
            return False

        return (rsi > 70 and macd < macd_signal) or (rsi < 30 and macd > macd_signal)
