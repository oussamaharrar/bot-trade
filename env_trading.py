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
        stop_loss: float | None = None,
        take_profit: float | None = None,
        max_loss_per_session: float | None = None,
    ):
        super().__init__()
        self.original_df = data.copy().reset_index(drop=True)
        self.df = add_strategy_features(self.original_df)
        self.initial_balance = initial_balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_loss_per_session = max_loss_per_session
        self.win_streak = 0
        self.loss_streak = 0
        self.risk_pct = 1.0

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
        self.risk_pct = 1.0
        return self._make_obs(self.index, self.usdt, self.coin), {}

    def step(self, action: int):
        done = False
        price = self.df.iloc[self.index].get("price", self.df.iloc[self.index].get("close", 0.0))

        risk = self.risk_pct
        if action == 1 and self.usdt > 0:  # BUY
            amount = (self.usdt * risk) / price
            self.coin += amount
            self.usdt -= amount * price
        elif action == 2 and self.coin > 0:  # SELL
            amount = self.coin * risk
            self.usdt += amount * price
            self.coin -= amount
        self.index += 1
        if self.index >= len(self.df) - 1:
            done = True
        new_price = self.df.iloc[self.index].get("price", self.df.iloc[self.index].get("close", 0.0))
        total_value = self.usdt + self.coin * new_price
        reward = total_value - self.prev_value
        self.prev_value = total_value

        if reward > 0:
            self.win_streak += 1
            self.loss_streak = 0
        elif reward < 0:
            self.loss_streak += 1
            self.win_streak = 0

        if self.win_streak >= 2:
            self.risk_pct = min(1.0, self.risk_pct + 0.1)
        elif self.loss_streak >= 2:
            self.risk_pct = max(0.1, self.risk_pct - 0.1)

        if self.stop_loss and total_value <= self.initial_balance * (1 - self.stop_loss):
            done = True
        if self.take_profit and total_value >= self.initial_balance * (1 + self.take_profit):
            done = True
        if self.max_loss_per_session and total_value <= self.initial_balance * (1 - self.max_loss_per_session):
            done = True

        obs = self._make_obs(self.index, self.usdt, self.coin)
        info = {"total_value": total_value, "timestamp": self.df.iloc[self.index]["timestamp"]}
        return obs, float(reward), done, False, info

