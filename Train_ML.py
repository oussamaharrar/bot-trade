
import logging
logging.basicConfig(level=logging.INFO)
import os
import json
from datetime import datetime
import pandas as pd
import yaml
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
from config.env_trading import TradingEnv

CONFIG_PATH = os.path.join('config', 'config.yaml')
if os.path.exists(CONFIG_PATH):     
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
    RL_CONFIG = CONFIG.get('rl', {})
else:
    RL_CONFIG = {}

DATA_PATH = r"training_dataset_full.csv"
AGENT_DIR = "agents"
os.makedirs(AGENT_DIR, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "price" in df.columns and "close" not in df.columns:
        df.rename(columns={"price": "close"}, inplace=True)
    df["volume"] = 1.0
    return df

def main(episodes: int = 10, patience: int = 3, agent_type: str | None = None, policy: str | None = None):
    df = load_data(DATA_PATH)

    agent_type = agent_type or RL_CONFIG.get('agent_type', 'PPO')
    policy = policy or RL_CONFIG.get('policy', 'MlpPolicy')

    cls_map = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN}
    ModelCls = cls_map.get(agent_type, PPO)

    stages = [
        {'use_indicators': False, 'max_steps': 10000},
        {'use_indicators': True, 'max_steps': 10000},
    ]

    env = DummyVecEnv([lambda: TradingEnv(df, **stages[0])])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    model = ModelCls(policy, env, verbose=1)

    best_reward = -np.inf
    no_improve = 0
    model_path = None
    rewards = []
    total_timesteps = 0
    stage_switch = episodes // len(stages) if episodes > 1 else 1
    current_stage = 0

    for ep in range(episodes):
        if ep and ep % stage_switch == 0 and current_stage < len(stages) - 1:
            current_stage += 1
            env = DummyVecEnv([lambda: TradingEnv(df, **stages[current_stage])])
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            model.set_env(env)

        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        total_timesteps += 10000

        obs = env.reset()
        done = False
        total_reward = 0.0
        step_counter = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(np.array([action]))
            reward = reward[0]
            total_reward += reward
            step_counter += 1
            if step_counter % 500 == 0:
                logging.info(f"[EP {ep+1} STEP {step_counter}] reward so far: {total_reward:.2f}")

        rewards.append(float(total_reward))
        logging.info(f"Episode {ep+1}: total_reward={total_reward:.2f}, best_reward={best_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            no_improve = 0
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(AGENT_DIR, f"{agent_type.lower()}_{ts}.zip")
            model.save(model_path)
            # حفظ نسخة مضغوطة من بيئة التطبيع
            env.save(os.path.join(AGENT_DIR, f"{agent_type.lower()}_{ts}_vecnorm.pkl"))

            with open(os.path.join(AGENT_DIR, "best_model.txt"), "w") as f:
                f.write(model_path)
            logging.info(f"Episode {ep+1}: improved reward {total_reward:.2f}. Model saved to {model_path}")
        else:
            no_improve += 1
            logging.info(f"Episode {ep+1}: reward {total_reward:.2f} (no improvement)")

        if no_improve >= patience:
            logging.info("Early stopping due to no reward improvement")
            break

    os.makedirs('reports', exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward')
        plt.savefig(os.path.join('reports', 'rl_reward_curve.png'))
        plt.close()
    except Exception:
        pass

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
    memory['rl_agent_type'] = agent_type
    memory['rl_last_train_steps'] = total_timesteps
    memory['rl_last_train_reward'] = float(best_reward)
    os.makedirs('memory', exist_ok=True)
    with open(mem_path, 'w') as m:
        json.dump(memory, m, indent=2)

if __name__ == "__main__":
    main()
