import os
import json
from datetime import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_trading import TradingEnv

DATA_PATH = "training_dataset.csv"
AGENT_DIR = "agents"
EPISODES = 50
PATIENCE = 5
os.makedirs(AGENT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.rename(columns={"price": "close"}, inplace=True)
    df["volume"] = 1.0
    return df


def evaluate(model: PPO, df: pd.DataFrame) -> float:
    """Run the model on the given dataframe and return the total reward."""
    env = TradingEnv(df)
    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
    final_value = info.get("total_value", env.prev_value)
    return final_value - env.initial_balance


def main():
    df = load_data(DATA_PATH)
    train_env = DummyVecEnv([lambda: TradingEnv(df)])
    eval_df = df.copy()

    model = PPO("MlpPolicy", train_env, verbose=1)

    best_reward = -float("inf")
    best_path_file = os.path.join(AGENT_DIR, "best_model.txt")
    if os.path.exists(best_path_file):
        try:
            with open(best_path_file, "r") as f:
                prev_path = f.read().strip()
            if os.path.exists(prev_path):
                prev_model = PPO.load(prev_path)
                best_reward = evaluate(prev_model, eval_df)
        except Exception:
            best_reward = -float("inf")

    no_improve = 0
    for ep in range(1, EPISODES + 1):
        model.learn(total_timesteps=len(df) - 1, reset_num_timesteps=False)
        reward = evaluate(model, eval_df)
        if reward > best_reward:
            best_reward = reward
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(AGENT_DIR, f"ppo_{ts}.zip")
            model.save(model_path)
            with open(best_path_file, "w") as f:
                f.write(model_path)
            print(f"Episode {ep}: reward {reward:.2f} -> new best model saved")
            no_improve = 0
        else:
            no_improve += 1
            print(f"Episode {ep}: reward {reward:.2f} (no improvement)")
            if no_improve >= PATIENCE:
                print(f"No improvement for {PATIENCE} episodes, stopping early")
                break

    mem_path = os.path.join('memory', 'memory.json')
    memory = {}
    if os.path.exists(mem_path):
        try:
            with open(mem_path, 'r') as m:
                memory = json.load(m)
        except Exception:
            memory = {}
    if os.path.exists(best_path_file):
        with open(best_path_file, 'r') as f:
            memory['rl_policy'] = os.path.basename(f.read().strip())
    memory['last_rl_reward'] = best_reward
    os.makedirs('memory', exist_ok=True)
    with open(mem_path, 'w') as m:
        json.dump(memory, m, indent=2)


if __name__ == "__main__":
    main()
