import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config.env_trading import TradingEnv

# ==== الإعدادات ====
AGENT_NAME = "ppo_main"   # عدّل الاسم حسب النموذج المطلوب
DATA_PATH = "training_dataset_full.feather"
AGENT_DIR = "agents"
LOG_FILE = os.path.join(AGENT_DIR, f"{AGENT_NAME}_train_log.csv")
N_STEPS = 100_000  # عدد الخطوات الجديدة
POLICY = "MlpPolicy"  # لا تغيّرها عادة

os.makedirs(AGENT_DIR, exist_ok=True)

# ==== تحميل البيانات ====
df = pd.read_feather(DATA_PATH)
env = DummyVecEnv([lambda: TradingEnv(df, use_indicators=True)])

# ==== تحميل البيئة المطبّعة والنموذج السابق ====
vecnorm_path = os.path.join(AGENT_DIR, f"{AGENT_NAME}_vecnorm.pkl")
model_path = os.path.join(AGENT_DIR, f"{AGENT_NAME}.zip")

if os.path.exists(vecnorm_path) and os.path.exists(model_path):
    env = VecNormalize.load(vecnorm_path, env)
    model = PPO.load(model_path, env=env)
    print(f"[INFO] Loaded existing agent {AGENT_NAME}")
else:
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    model = PPO(POLICY, env, verbose=1)
    print(f"[INFO] Started new agent {AGENT_NAME}")

# ==== مواصلة التدريب ====
results = []
best_reward = -float('inf')
best_model_path = None

if os.path.exists(os.path.join(AGENT_DIR, "best_model.txt")):
    with open(os.path.join(AGENT_DIR, "best_model.txt")) as f:
        best_model_path = f.read().strip()

try:
    for session in range(1, 101):  # 100 جلسة كحد أقصى (يمكنك تغييره)
        print(f"========== Session {session} ==========")
        model.learn(total_timesteps=N_STEPS, reset_num_timesteps=False)
        # التقييم السريع على نفس البيئة
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
        print(f"[INFO] Session {session}: total_reward={total_reward:.2f}")

        # حفظ النتائج
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_info = {
            "session": session,
            "timestamp": ts,
            "reward": float(total_reward),
            "steps": N_STEPS * session
        }
        results.append(session_info)

        # حفظ سجل الأداء
        pd.DataFrame(results).to_csv(LOG_FILE, index=False)

        # إذا كان هناك تحسن في الأداء
        if total_reward > best_reward:
            best_reward = total_reward
            # حفظ النموذج وبيئة VecNormalize
            model_out = os.path.join(AGENT_DIR, f"{AGENT_NAME}_{ts}.zip")
            vec_out = os.path.join(AGENT_DIR, f"{AGENT_NAME}_{ts}_vecnorm.pkl")
            model.save(model_out)
            env.save(vec_out)
            # تحديث مؤشر أفضل نموذج
            with open(os.path.join(AGENT_DIR, "best_model.txt"), "w") as f:
                f.write(model_out)
            print(f"[BEST] New best model saved: {model_out}")

except KeyboardInterrupt:
    print("[STOP] Training interrupted by user.")

print(f"[DONE] Training log saved to {LOG_FILE}")
