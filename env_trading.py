import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from typing import List, Tuple

from strategy_features import add_strategy_features

class TradingEnv(Env):
    """Simple trading environment for RL based on historical OHLCV data."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 1000.0,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        max_session_loss_pct: float | None = None,
        risk_pct_hook=None,
    ):
        super().__init__()
        self.original_df = data.copy().reset_index(drop=True)
        self.df = add_strategy_features(self.original_df)
        self.initial_balance = initial_balance
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_session_loss_pct = max_session_loss_pct
        self.risk_pct_hook = risk_pct_hook
        self.current_risk_pct = 1.0
        self.win_streak = 0
        self.loss_streak = 0
        self.entry_price = None

        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        # observation: indicators + price change + position state
        sample_obs = self._make_obs(0, 0.0, 0.0)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(sample_obs),),
            dtype=np.float32,
        )

        self.reset()

    def _make_obs(self, idx: int, usdt: float, coin: float) -> np.ndarray:
        row = self.df.iloc[idx]
        obs = [
            row.get("price", row.get("close", 0.0)),
            row.get("price_change", 0.0),
            row.get("rsi_14", 0.0),
            row.get("macd", 0.0),
            row.get("macd_signal", 0.0),
            row.get("macd_hist", 0.0),
            row.get("bollinger_upper_diff", 0.0),
            row.get("bollinger_lower_diff", 0.0),
            usdt / self.initial_balance,
            (coin * row.get("price", row.get("close", 0.0))) / self.initial_balance,
        ]
        return np.array(obs, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.index = 0
        self.usdt = self.initial_balance
        self.coin = 0.0
        self.prev_value = self.initial_balance
        self.win_streak = 0
        self.loss_streak = 0
        self.current_risk_pct = (
            1.0 if self.risk_pct_hook is None else self.risk_pct_hook(0, 0)
        )
        self.entry_price = None
        return self._make_obs(self.index, self.usdt, self.coin), {}

    def step(self, action: int):
        done = False
        price = self.df.iloc[self.index].get(
            "price", self.df.iloc[self.index].get("close", 0.0)
        )

        risk = self.current_risk_pct
        if action == 1 and self.usdt > 0:  # BUY
            amount_usdt = self.usdt * risk
            amount = amount_usdt / price
            self.coin += amount
            self.usdt -= amount_usdt
            if self.entry_price is None:
                self.entry_price = price
        elif action == 2 and self.coin > 0:  # SELL
            amount_coin = self.coin * risk
            self.usdt += amount_coin * price
            self.coin -= amount_coin
            if self.coin == 0:
                self.entry_price = None

        self.index += 1
        if self.index >= len(self.df) - 1:
            done = True

        new_price = self.df.iloc[self.index].get(
            "price", self.df.iloc[self.index].get("close", 0.0)
        )

        if self.coin > 0 and self.entry_price is not None:
            if (
                self.stop_loss_pct is not None
                and new_price <= self.entry_price * (1 - self.stop_loss_pct)
            ):
                self.usdt += self.coin * new_price
                self.coin = 0.0
                self.entry_price = None
            elif (
                self.take_profit_pct is not None
                and new_price >= self.entry_price * (1 + self.take_profit_pct)
            ):
                self.usdt += self.coin * new_price
                self.coin = 0.0
                self.entry_price = None

        total_value = self.usdt + self.coin * new_price

        if (
            self.max_session_loss_pct is not None
            and total_value <= self.initial_balance * (1 - self.max_session_loss_pct)
        ):
            done = True

        reward = total_value - self.prev_value
        self.prev_value = total_value

        if reward > 0:
            self.win_streak += 1
            self.loss_streak = 0
        elif reward < 0:
            self.loss_streak += 1
            self.win_streak = 0
        else:
            self.win_streak = 0
            self.loss_streak = 0

        if self.risk_pct_hook:
            self.current_risk_pct = self.risk_pct_hook(
                self.win_streak, self.loss_streak
            )
        else:
            self.current_risk_pct = 1.0

        obs = self._make_obs(self.index, self.usdt, self.coin)
        info = {
            "total_value": total_value,
            "timestamp": self.df.iloc[self.index]["timestamp"],
            "price": new_price,
        }
        return obs, float(reward), done, False, info

