import os
import json
from datetime import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_trading import TradingEnv

DATA_PATH = "training_dataset.csv"
AGENT_DIR = "agents"
os.makedirs(AGENT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.rename(columns={"price": "close"}, inplace=True)
    df["volume"] = 1.0
    return df


def main():
    df = load_data(DATA_PATH)
    env = DummyVecEnv([lambda: TradingEnv(df)])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(AGENT_DIR, f"ppo_{ts}.zip")
    model.save(model_path)
    with open(os.path.join(AGENT_DIR, "best_model.txt"), "w") as f:
        f.write(model_path)
    print(f"Model saved to {model_path}")

    mem_path = os.path.join('memory', 'memory.json')
    memory = {}
    if os.path.exists(mem_path):
        try:
            with open(mem_path, 'r') as m:
                memory = json.load(m)
        except Exception:
            memory = {}
    memory['rl_policy'] = os.path.basename(model_path)
    os.makedirs('memory', exist_ok=True)
    with open(mem_path, 'w') as m:
        json.dump(memory, m, indent=2)


if __name__ == "__main__":
    main()
