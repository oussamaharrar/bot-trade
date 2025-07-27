import os
import json
import pandas as pd
from stable_baselines3 import PPO

from env_trading import TradingEnv
from market_data import fetch_ohlcv
from results_logger import simulate_wallet

AGENT_DIR = "agents"


def load_best_model():
    path_file = os.path.join(AGENT_DIR, "best_model.txt")
    if os.path.exists(path_file):
        with open(path_file, "r") as f:
            path = f.read().strip()
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Best model not found. Train the RL agent first.")


def main(model_path: str | None = None):
    if model_path is None:
        model_path = load_best_model()

    model = PPO.load(model_path)
    df = fetch_ohlcv()
    df.rename(columns={"close": "price"}, inplace=True)
    df["volume"] = 1.0
    env = TradingEnv(df)

    obs, _ = env.reset()
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        ts = info.get("timestamp")
        price = info.get("price", df.iloc[env.index]["price"] if env.index < len(df) else df.iloc[-1]["price"])
        signal = {0: "HOLD", 1: "BUY", 2: "SELL"}[int(action)]
        actions.append((ts, price, signal))

    logs, final_val = simulate_wallet(actions)
    print(f"Final portfolio value: {final_val:.2f} USDT")

    df_logs = pd.DataFrame(
        logs,
        columns=[
            "timestamp",
            "price",
            "signal",
            "usdt",
            "coin_value",
            "total_value",
            "note",
            "pnl",
            "status",
        ],
    )
    if not df_logs.empty:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
        df_logs["date"] = df_logs["timestamp"].dt.date
        avg_daily_pnl = df_logs.groupby("date")["pnl"].sum().mean()
        roll_max = df_logs["total_value"].cummax()
        drawdown = (df_logs["total_value"] - roll_max) / roll_max
        rolling_drawdown = float(drawdown.iloc[-1])
    else:
        avg_daily_pnl = 0.0
        rolling_drawdown = 0.0

    # update memory
    mem_path = os.path.join('memory', 'memory.json')
    memory = {}
    if os.path.exists(mem_path):
        try:
            with open(mem_path, 'r') as f:
                memory = json.load(f)
        except Exception:
            memory = {}
    memory['last_rl_reward'] = final_val - env.initial_balance
    memory['last_rl_total_value'] = final_val
    memory['rl_policy'] = os.path.basename(model_path)
    memory['rolling_drawdown'] = rolling_drawdown
    memory['avg_daily_pnl'] = avg_daily_pnl
    os.makedirs('memory', exist_ok=True)
    with open(mem_path, 'w') as f:
        json.dump(memory, f, indent=2)

    # Append new run to dataset for continual learning
    dataset = os.path.join(os.path.dirname(__file__), "training_dataset.csv")
    res_df = pd.DataFrame(actions, columns=["timestamp", "price", "signal"])
    res_df.to_csv(dataset, mode="a", index=False, header=False)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
