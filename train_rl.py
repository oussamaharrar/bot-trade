import os
import json
from datetime import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

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


def main(episodes: int = 10, patience: int = 3):
    df = load_data(DATA_PATH)
    env = DummyVecEnv([lambda: TradingEnv(df)])

    model = PPO("MlpPolicy", env, verbose=1)

    best_reward = -np.inf
    no_improve = 0
    model_path = None

    for ep in range(episodes):
        model.learn(total_timesteps=10000, reset_num_timesteps=False)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(int(action))
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward
            no_improve = 0
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(AGENT_DIR, f"ppo_{ts}.zip")
            model.save(model_path)
            with open(os.path.join(AGENT_DIR, "best_model.txt"), "w") as f:
                f.write(model_path)
            print(f"Episode {ep+1}: improved reward {total_reward:.2f}. Model saved to {model_path}")
        else:
            no_improve += 1
            print(f"Episode {ep+1}: reward {total_reward:.2f} (no improvement)")

        if no_improve >= patience:
            print("Early stopping due to no reward improvement")
            break

    mem_path = os.path.join('memory', 'memory.json')
    memory = {}
    if os.path.exists(mem_path):
        try:
            with open(mem_path, 'r') as m:
                memory = json.load(m)
        except Exception:
            memory = {}
    if model_path:
        memory['rl_policy'] = os.path.basename(model_path)
    os.makedirs('memory', exist_ok=True)
    with open(mem_path, 'w') as m:
        json.dump(memory, m, indent=2)


if __name__ == "__main__":
    main()
