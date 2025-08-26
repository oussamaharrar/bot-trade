#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_rl.py — SB3 PPO trainer with session isolation, playlist runs, auto-resume, and live progress.
متوافق مع Windows / Python 3.10+ / Stable-Baselines3.

الجديد اليوم:
- تقارير فورية عن العتاد (CUDA/GPUs/CPU threads) + طباعة الجهاز الفعلي للموديل.
- شريط تقدّم لتحميل الملفات + تقدّم التعلم (SB3 progress_bar) + طباعة دورية كل N ثوانٍ.
- تسجيل دوري إلى results/step_log.csv (timesteps, ep_rew_mean, FPS, GPU Mem...).
- جاهز للتكامل مع tools/analyze_risk_and_training.py تلقائيًا بعد كل ملف (إن وُجد) + يعمل بانسجام مع ai_core/config/results.
- إصلاح تمرير بارامترات غير مدعومة للبيئة.
- **جديد**: استئناف صحيح لـ VecNormalize + حفظ أفضل نقطة (BestCheckpoint) تلقائيًا.
"""

import os
import gc
import sys
import json
import glob
import time
import argparse
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional

import pyarrow as pa
import pyarrow.ipc as pa_ipc


import numpy as np
import pandas as pd

# PyTorch / CUDA
import torch
# اضبط خيوط بايثورش في العملية الرئيسية (قيمة آمنة)
torch.set_num_threads(6)
torch.set_num_interop_threads(1) 

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from config.strategy_features import add_strategy_features

import multiprocessing as mp
import logging

from functools import partial

# على Windows نجبر spawn مبكرًا (لو لم يكن مضبوطًا)
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


try:
    import gymnasium as gym
except Exception:
    import gym


# قلّل تداخل الخيوط داخل كل عملية (مهم جداً مع SubprocVecEnv)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

# اخفض ضوضاء لوغر إدارة المخاطر أثناء التدريب
logging.getLogger("config.risk_manager").setLevel(logging.WARNING)

# بيئتنا
from config.env_trading import TradingEnv  # يعيد (obs, reward, terminated, truncated, info)

# tqdm (اختياري لعرض التقدّم)
try:
    from tqdm import tqdm
except Exception:  # fallback إذا لم تتوفر tqdm
    def tqdm(x, **kwargs):
        return x


# ============================================================
# أدوات عامة
# ============================================================

# Top-level factory to avoid Windows spawn/closure issues
def _init_trading_env(df, symbol, frame):
    # بيانات المؤشرات حُسِبت مسبقًا، لذا نمرر use_indicators=False
    return TradingEnv(data=df, symbol=symbol, frame=frame, use_indicators=False)


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def list_feather_files(data_dir: str, frame: str, symbol: str) -> List[str]:
    pattern1 = os.path.join(data_dir, frame, f"*{symbol}*{frame}*.feather")
    pattern2 = os.path.join(data_dir, frame, f"*{symbol}*.feather")
    files = sorted(set(glob.glob(pattern1) + glob.glob(pattern2)))
    return files


def load_feather_numeric(path: str, safe: bool = False) -> pd.DataFrame:
    t0 = time.time()
    # نقرأ ملف Feather كسلسلة record batches لعرض شريط تقدّم
    try:
        with pa_ipc.open_file(path) as reader:
            total_batches = reader.num_record_batches
            if total_batches is None or total_batches <= 0:
                # fallback: ملف بدون batches واضحة
                df = pd.read_feather(path)
                num_df = df.select_dtypes(include=[np.number]).copy()
            else:
                from tqdm import tqdm  # موجودة أصلاً
                parts = []
                pbar = tqdm(total=total_batches, desc=f"[read] {os.path.basename(path)}", unit="batch", leave=True)
                for i in range(total_batches):
                    batch = reader.get_batch(i)                    # pa.RecordBatch
                    tbl = pa.Table.from_batches([batch])           # pa.Table
                    part = tbl.to_pandas(types_mapper=None)        # -> pandas
                    part = part.select_dtypes(include=[np.number]) # نحتفظ بالرقمية فقط
                    # downcast لتقليل الذاكرة
                    for c in part.columns:
                        if pd.api.types.is_float_dtype(part[c]) or pd.api.types.is_integer_dtype(part[c]):
                            part[c] = part[c].astype(np.float32, copy=False)
                    parts.append(part)
                    pbar.update(1)
                pbar.close()
                num_df = pd.concat(parts, ignore_index=True)
    except Exception:
        # fallback آمن إذا pyarrow فشل لأي سبب
        df = pd.read_feather(path)
        num_df = df.select_dtypes(include=[np.number]).copy()
        for c in num_df.columns:
            if pd.api.types.is_float_dtype(num_df[c]) or pd.api.types.is_integer_dtype(num_df[c]):
                num_df[c] = num_df[c].astype(np.float32, copy=False)

    # تنظيف NaN/Inf
    num_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_df.ffill(inplace=True)
    num_df.fillna(0.0, inplace=True)
    if safe and (num_df.isna().sum().sum() > 0):
        raise ValueError("NaN detected after cleaning")

    dt = time.time() - t0
    print(f"[📄] Loaded {os.path.basename(path)} shape={num_df.shape} in {dt:.2f}s")
    return num_df



def adjust_batch(n_envs: int, n_steps: int, batch_size: int) -> int:
    max_batch = n_envs * n_steps
    batch = min(batch_size, max_batch)
    batch = (batch // n_envs) * n_envs  # قابل للقسمة على n_envs
    return max(n_envs, batch)


def device_name(index: int) -> str:
    if torch.cuda.is_available():
        if index is not None and index >= 0 and index < torch.cuda.device_count():
            return f"cuda:{index}"
        return "cuda:0"
    return "cpu"


def print_device_report(chosen_index: Optional[int]):
    print("========== DEVICE REPORT ==========")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device_count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            print(f"  [{i}] {name}")
        if chosen_index is not None and 0 <= chosen_index < torch.cuda.device_count():
            torch.cuda.set_device(chosen_index)
        current = torch.cuda.current_device()
        print(f"Selected device index: {current} -> {torch.cuda.get_device_name(current)}")
    else:
        print("CUDA not available — running on CPU.")
    try:
        print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    except Exception:
        pass
    print("===================================")


# ============================================================
# Callbacks
# ============================================================

class StatusLoggerCallback(BaseCallback):
    """يطبع حالة التدريب دوريًا ويكتب إلى step_log.csv كل (log_every_steps) أو (every_sec)."""

    def __init__(self, step_log_path: str, device_str: str, every_sec: int = 10, log_every_steps: int = 10_000):
        super().__init__()
        self.step_log_path = step_log_path
        self.device_str = device_str
        self.every_sec = max(1, int(every_sec))
        self.log_every_steps = max(1, int(log_every_steps))
        self.t0 = time.time()
        self._last_print_t = self.t0
        self._last_step = 0
        # ترويسة ملف السجل إن لم يكن موجودًا
        if not os.path.exists(self.step_log_path):
            ensure_dir(os.path.dirname(self.step_log_path))
            pd.DataFrame([
                {"ts": "iso", "timesteps": 0, "ep_rew_mean": 0.0, "fps": 0.0,
                 "device": self.device_str, "gpu_alloc_mb": 0, "gpu_reserved_mb": 0}
            ]).iloc[0:0].to_csv(self.step_log_path, index=False, encoding="utf-8")

    def _write_row(self, ep_rew_mean: Optional[float] = None):
        now = time.time()
        steps = int(self.model.num_timesteps)
        dt = max(1e-9, now - self.t0)
        fps = steps / dt
        # GPU stats
        alloc_mb = reserved_mb = 0
        if self.device_str.startswith("cuda") and torch.cuda.is_available():
            try:
                alloc_mb = int(torch.cuda.memory_allocated() / (1024**2))
                reserved_mb = int(torch.cuda.memory_reserved() / (1024**2))
            except Exception:
                pass
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "timesteps": steps,
            "ep_rew_mean": float(ep_rew_mean) if ep_rew_mean is not None else np.nan,
            "fps": round(fps, 2),
            "device": self.device_str,
            "gpu_alloc_mb": alloc_mb,
            "gpu_reserved_mb": reserved_mb,
        }
        # append
        pd.DataFrame([row]).to_csv(self.step_log_path, mode="a", header=False, index=False, encoding="utf-8")
        # print
        msg = (
            f"[⏱️ {row['ts']}] steps={steps:,} | fps≈{row['fps']} | ep_rew_mean={row['ep_rew_mean']} | "
            f"dev={self.device_str} | vram={alloc_mb}/{reserved_mb} MB"
        )
        print(msg)

    def _on_step(self) -> bool:
        now = time.time()
        steps = int(self.model.num_timesteps)
        if (steps - self._last_step) >= self.log_every_steps or (now - self._last_print_t) >= self.every_sec:
            # حاول استخراج ep_rew_mean من اللوغر إن وُجد
            ep_rew_mean = None
            try:
                logs = self.model.logger.get_log_dict()
                if "rollout/ep_rew_mean" in logs:
                    ep_rew_mean = logs["rollout/ep_rew_mean"]
                elif "train/ep_rew_mean" in logs:
                    ep_rew_mean = logs["train/ep_rew_mean"]
            except Exception:
                pass
            self._write_row(ep_rew_mean)
            self._last_print_t = now
            self._last_step = steps
        return True


class BestCheckpointCallback(BaseCallback):
    """يحفظ أفضل نموذج بناءً على metric (افتراضيًا rollout/ep_rew_mean)."""

    def __init__(self, save_dir: str, vecnorm_ref: Optional[VecNormalize], metric: str = "rollout/ep_rew_mean"):
        super().__init__()
        self.save_dir = ensure_dir(save_dir)
        self.vecnorm_ref = vecnorm_ref
        self.metric = metric
        self.best_val = -np.inf
        self.best_path_model = os.path.join(self.save_dir, "deep_rl_best.zip")
        self.best_path_vecnorm = os.path.join(self.save_dir, "vecnorm_best.pkl")
        self.meta_path = os.path.join(self.save_dir, "best_ckpt.json")

    def _on_step(self) -> bool:
        try:
            logs = self.model.logger.get_log_dict()
            val = logs.get(self.metric, None)
            if val is None:
                # fallback لبعض النسخ
                val = logs.get("train/ep_rew_mean", None)
            if val is None or not np.isfinite(val):
                return True
            if float(val) > float(self.best_val):
                self.best_val = float(val)
                # Save model
                self.model.save(self.best_path_model)
                # Save vecnorm stats if available
                try:
                    if self.vecnorm_ref is not None:
                        self.vecnorm_ref.save(self.best_path_vecnorm)
                except Exception:
                    pass
                # meta
                meta = {
                    "metric": self.metric,
                    "value": float(self.best_val),
                    "timesteps": int(self.model.num_timesteps),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                try:
                    with open(self.meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception:
            pass
        return True


# ============================================================
# تدريب ملف واحد (جلسة واحدة)
# ============================================================

from functools import partial
import os

def init_env_worker(df, symbol, frame):
    # حدّ خيوط المكتبات داخل كل عامل لتجنّب تضخيم الخيوط (oversubscription)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    # نمرّر use_indicators=False لأننا حسبنا المؤشرات مسبقًا في العملية الرئيسية
    return TradingEnv(data=df, symbol=symbol, frame=frame, use_indicators=False)

def build_env_fns(n_envs: int, df: pd.DataFrame, symbol: str, frame: str) -> List:
    def make_env(rank):
        def _init():
            # تحديد الخيوط لكل عامل لتجنّب تشبع الـ CPU
            import os, torch, traceback
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            try:
                env = TradingEnv(data=df, symbol=symbol, frame=frame, use_indicators=False)
            except Exception as e:
                # اطبع ستاك تريس واضح من العامل ثم أعد الرمي لإيقاف العامل بدل إرجاع None
                traceback.print_exc()
                raise
            return env
        return _init
    return [make_env(i) for i in range(n_envs)]




def maybe_load_vecnorm(vec_env: VecNormalize, vecnorm_best_path: str, vecnorm_path: str) -> VecNormalize:
    """تحميل إحصاءات VecNormalize إن وجدت (الأفضل ثم العادي)."""
    loaded = False
    if os.path.isfile(vecnorm_best_path):
        try:
            vec_env = VecNormalize.load(vecnorm_best_path, vec_env)
            loaded = True
            print(f"[♻️] Loaded VecNormalize stats: {vecnorm_best_path}")
        except Exception as e:
            print(f"[WARN] Failed to load vecnorm_best: {e}")
    if (not loaded) and os.path.isfile(vecnorm_path):
        try:
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            loaded = True
            print(f"[♻️] Loaded VecNormalize stats: {vecnorm_path}")
        except Exception as e:
            print(f"[WARN] Failed to load vecnorm: {e}")
    # تأكيد وضع التدريب
    try:
        vec_env.training = True
        vec_env.clip_obs = 10.0
    except Exception:
        pass
    return vec_env


def train_one_file( 
       
        file_path: str,
        symbol: str,
        frame: str,
        args,
        agent_dir: str,
        results_dir: str,
        logs_dir: str,
        memory_path: str,
        vecnorm_path: str,
        resume_auto: str = "latest",
):
    print(f"[📂] Loading feather: {file_path}")
    df = load_feather_numeric(file_path, safe=args.safe)

    # إعداد البيئات
    print("[🧪] Creating vectorized envs (SubprocVecEnv)...")
    env_fns = build_env_fns(args.n_envs, df, symbol, frame)
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # حمل إحصاءات VecNormalize قبل تحميل/بناء النموذج
    vecnorm_best_path = os.path.join(agent_dir, "vecnorm_best.pkl")
    vec_env = maybe_load_vecnorm(vec_env, vecnorm_best_path, vecnorm_path)

    # ضبط n_steps/batch
    n_steps = min(len(df) // max(1, args.n_envs), args.n_steps)
    n_steps = max(n_steps, 128)
    batch_size = adjust_batch(args.n_envs, n_steps, args.batch_size)
    print(f"[⚙️] n_envs={args.n_envs} | n_steps={n_steps} | batch_size={batch_size}")
    import torch
    # GPU / TF32
    print_device_report(args.device if args.device is not None else -1)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        if args.device is not None and 0 <= args.device < torch.cuda.device_count():
            torch.cuda.set_device(args.device)

    device_str = device_name(args.device)

    # نموذج PPO
    policy_kwargs = dict(net_arch=[256, 256])
    # Enable gSDE only for continuous action spaces (Box); disable for Discrete.
    try:
        from gymnasium import spaces
    except Exception:
        from gym import spaces  # fallback for older SB3/gym

    
        # تحديد نوع الأكشن بشكل متوافق مع كل الإصدارات
    action_space = getattr(vec_env, "single_action_space", None)
    if action_space is None:
        action_space = getattr(vec_env, "action_space", None)
    # محاولة أخيرة عبر get_attr من أول بيئة
    if action_space is None:
        try:
            action_space = vec_env.get_attr("action_space")[0]
        except Exception:
            pass
    is_continuous = isinstance(action_space, gym.spaces.Box)

    # فعّل gSDE فقط إذا الأكشن continuous وفقط إن طلبت من CLI
    use_sde = bool(args.use_sde and is_continuous)

    print(f"[🧠] Building PPO on device: {device_str} "
          f"(use_sde={use_sde}, action_space={type(action_space).__name__})")

    if args.use_sde and not is_continuous:
        print("[WARN] use-sde requested but action space is Discrete; disabling gSDE.")



    ensure_dir(agent_dir); ensure_dir(results_dir); ensure_dir(logs_dir)

    # الاستئناف التلقائي (الأفضل ثم العادي)
    model = None
    model_best_path = os.path.join(agent_dir, "deep_rl_best.zip")
    model_path = os.path.join(agent_dir, "deep_rl.zip")
    load_candidates = [model_best_path, model_path]
    if resume_auto:
        for cand in load_candidates:
            if os.path.exists(cand):
                try:
                    print(f"[♻️] Auto-resume: loading {cand}")
                    model = PPO.load(cand, device=device_str, print_system_info=False)
                    model.set_env(vec_env)
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load model from {cand}: {e}")

    if model is None:
        import torch as th
        policy_kwargs = dict(
            net_arch=dict(pi=[2048, 2048, 1024], vf=[2048, 2048, 1024]),
            activation_fn=th.nn.SiLU,
            ortho_init=False,
        )
        model = PPO(
            "MlpPolicy",
            vec_env,
            device=device_str,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=args.epochs,
            learning_rate=args.lr,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=use_sde,
            sde_sample_freq=(4 if use_sde else -1),
            policy_kwargs=policy_kwargs,
            verbose=0,
        )


    # Checkpoints (خطوات مطلقة)
    ckpt_every = max(1, int(args.checkpoint_every))
    ckpt_callback = CheckpointCallback(
        save_freq=ckpt_every,
        save_path=agent_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Status logger (طباعة/تسجيل دوري)
    step_log = os.path.join(results_dir, "step_log.csv")
    status_cb = StatusLoggerCallback(step_log_path=step_log, device_str=device_str,
                                     every_sec=args.print_every_sec, log_every_steps=args.log_every_steps)

    # Best checkpoint saver
    best_cb = BestCheckpointCallback(save_dir=agent_dir, vecnorm_ref=vec_env)

    # بدء التدريب
    t0 = time.time()
    print(f"[🚀] Training for total_steps={args.total_steps:,} ...")
    from stable_baselines3.common.callbacks import BaseCallback
    import numpy as np, torch

    class StatusPrinterCallback(BaseCallback):
        def __init__(self, every_steps: int = 2048, name: str = "STAT"):
            super().__init__()
            self.every_steps = int(every_steps)
            self.name = name

        def _on_step(self) -> bool:
            if self.model.num_timesteps % self.every_steps != 0:
                return True
            try:
                usdt  = np.array(self.model.env.get_attr("usdt"), dtype=float)
                value = np.array(self.model.env.get_attr("prev_value"), dtype=float)
                dd    = np.array(self.model.env.get_attr("max_drawdown"), dtype=float)
                risk  = np.array(self.model.env.get_attr("risk_pct"), dtype=float)

                # تقدير PNL لحظي من آخر عنصرين في equity_curve
                eqs = self.model.env.get_attr("equity_curve")
                pnl = []
                for e in eqs:
                    if isinstance(e, list) and len(e) > 1:
                        pnl.append(e[-1] - e[-2])
                    else:
                        pnl.append(0.0)
                pnl = np.array(pnl, dtype=float)

                gpu_line = ""
                if torch.cuda.is_available():
                    dev_id = torch.cuda.current_device()
                    mem_mb = torch.cuda.memory_reserved(dev_id) / (1024**2)
                    max_mb = torch.cuda.get_device_properties(dev_id).total_memory / (1024**2)
                    gpu_line = f" | GPU{dev_id} mem≈{mem_mb:.0f}/{max_mb:.0f}MB"

                print(
                    f"[{self.name}] step={self.model.num_timesteps:,} "
                    f"USDT(mean)={usdt.mean():.2f}  EQUITY(mean)={value.mean():.2f}  "
                    f"PNL(mean)={pnl.mean():.2f}  DD(max)={dd.max():.3f}  RISK(mean)={risk.mean():.3f}"
                    f"{gpu_line}",
                    flush=True
                )
            except Exception as e:
                print(f"[{self.name}] warn: {e}", flush=True)
            return True

    live_cb = StatusPrinterCallback(every_steps=args.n_steps)
    model.learn(
        total_timesteps=int(args.total_steps),
        callback=[ckpt_callback, status_cb, best_cb, live_cb],
        progress_bar=True
    )

    dur = time.time() - t0

    # حفظ النموذج والـVecNormalize
    model.save(model_path)
    try:
        vec_env.save(vecnorm_path)
    except Exception:
        pass

    # سجل الجلسة
    train_log = os.path.join(results_dir, "train_log.csv")
    ensure_dir(os.path.dirname(train_log))
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "frame": frame,
        "file": os.path.basename(file_path),
        "steps": int(args.total_steps),
        "n_envs": args.n_envs,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "duration_sec": int(dur),
        "notes": ""
    }
    if os.path.exists(train_log):
        pd.DataFrame([row]).to_csv(train_log, mode="a", header=False, index=False, encoding="utf-8")
    else:
        pd.DataFrame([row]).to_csv(train_log, index=False, encoding="utf-8")

    # تحديث الذاكرة
    ensure_dir(os.path.dirname(memory_path))
    mem = {}
    if os.path.isfile(memory_path):
        try:
            mem = json.load(open(memory_path, "r", encoding="utf-8")) or {}
        except Exception:
            mem = {}
    key = f"{symbol}_{frame}"
    mem[key] = {
        "last_file": os.path.basename(file_path),
        "last_steps": int(args.total_steps),
        "last_train_ts": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        json.dump(mem, open(memory_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        print("[WARN] failed to save memory.json")

    # تحليل ما بعد التدريب (إن وُجد السكربت)
    if args.post_analyze:
        analyzer = os.path.join("tools", "analyze_risk_and_training.py")
        if os.path.isfile(analyzer):
            cmd = f"python {analyzer} --frame {frame} --agent_dir {agent_dir}"
            print(f"[🧠] Post-analyze: {cmd}")
            try:
                os.system(cmd)
            except Exception as e:
                print(f"[WARN] post-analyze failed: {e}")

    # إغلاق البيئات وتحرير الذاكرة
    try:
        vec_env.close()
    except Exception:
        pass
    del model, vec_env, df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# تشغيل Playlist
# ============================================================

def run_playlist(pl_path: str, args):
    import yaml
    with open(pl_path, "r", encoding="utf-8") as f:
        playlist = yaml.safe_load(f) or []

    for i, job in enumerate(playlist):
        symbol = job.get("symbol", args.symbol)
        frame = job.get("frame", args.frame)
        device = job.get("device", args.device)
        n_envs = int(job.get("n_envs", args.n_envs))
        n_steps = int(job.get("n_steps", args.n_steps))
        batch_size = int(job.get("batch_size", args.batch_size))
        epochs = int(job.get("epochs", args.epochs))
        total_steps = int(job.get("total_steps", args.total_steps))
        checkpoint_every = int(job.get("checkpoint_every", args.checkpoint_every))
        resume_auto = str(job.get("resume_auto", args.resume_auto)).lower() in ["1", "true", "yes", "latest"]

        class JArgs: ...
        jargs = JArgs()
        jargs.n_envs = n_envs
        jargs.n_steps = n_steps
        jargs.batch_size = batch_size
        jargs.epochs = epochs
        jargs.total_steps = total_steps
        jargs.checkpoint_every = checkpoint_every
        jargs.device = device
        jargs.lr = args.lr
        jargs.safe = args.safe
        jargs.resume_auto = resume_auto
        jargs.post_analyze = args.post_analyze
        jargs.print_every_sec = args.print_every_sec
        jargs.log_every_steps = args.log_every_steps

        print(f"================= PLAYLIST ITEM {i+1}/{len(playlist)} =================")
        print(f"symbol={symbol} | frame={frame} | device={device}")

        data_dir = args.data_dir
        files = list_feather_files(data_dir, frame, symbol)
        if not files:
            print(f"[WARN] No feather files found for {symbol} {frame} under {os.path.join(data_dir, frame)}")
            continue

        agent_dir = ensure_dir(os.path.join(args.agents_dir, symbol, frame))
        results_dir = ensure_dir(os.path.join(args.results_dir, symbol, frame))
        logs_dir = ensure_dir(os.path.join(args.logs_dir, symbol, frame))
        memory_path = os.path.join(args.memory_dir, "memory.json")
        vecnorm_path = os.path.join(agent_dir, "vecnorm.pkl")

        for fp in tqdm(files, desc=f"{symbol}-{frame}"):
            train_one_file(
                file_path=fp,
                symbol=symbol,
                frame=frame,
                args=jargs,
                agent_dir=agent_dir,
                results_dir=results_dir,
                logs_dir=logs_dir,
                memory_path=memory_path,
                vecnorm_path=vecnorm_path,
                resume_auto="latest" if resume_auto else "",
            )


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("train_rl")
    # مسارات أساسية
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--agents-dir", type=str, default="agents")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--logs-dir", type=str, default="logs")
    p.add_argument("--memory-dir", type=str, default="memory")

    # اختيار عام
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--frame", type=str, default="1m")

    # موارد
    p.add_argument("--device", type=int, default=0, help="CUDA device index, -1 for CPU")
    p.add_argument("--seed", type=int, default=42)

    # RL
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=65_536)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--checkpoint-every", type=int, default=1_000_000)
    p.add_argument("--use-sde", action="store_true")
    p.add_argument("--lr", type=float, default=3e-4)

    # سلوك
    p.add_argument("--resume-auto", type=str, default="latest", help="latest/''")
    p.add_argument("--safe", action="store_true")
    p.add_argument("--post-analyze", action="store_true", help="Run tools/analyze_risk_and_training.py after each file")

    # عرض حي
    p.add_argument("--print-every-sec", type=int, default=10, help="طباعة حالة التدريب كل N ثوانٍ")
    p.add_argument("--log-every-steps", type=int, default=10_000, help="تسجيل حالة التدريب كل N خطوة")

    # Playlist
    p.add_argument("--playlist", type=str, default="", help="Path to playlist.yaml")

    return p.parse_args()


def main():
    args = parse_args()

    # بذور
    if args.seed is not None and args.seed >= 0:
        set_random_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # وضع Playlist؟
    if args.playlist and os.path.isfile(args.playlist):
        run_playlist(args.playlist, args)
        print("[✅] Playlist finished.")
        return

    # تشغيل ملف/ملفات (رمز/فريم) مباشرة
    files = list_feather_files(args.data_dir, args.frame, args.symbol)
    if not files:
        print(f"[ERROR] No feather files found for {args.symbol} {args.frame} under {os.path.join(args.data_dir, args.frame)}")
        sys.exit(1)

    agent_dir = ensure_dir(os.path.join(args.agents_dir, args.symbol, args.frame))
    results_dir = ensure_dir(os.path.join(args.results_dir, args.symbol, args.frame))
    logs_dir = ensure_dir(os.path.join(args.logs_dir, args.symbol, args.frame))
    memory_path = os.path.join(args.memory_dir, "memory.json")
    vecnorm_path = os.path.join(agent_dir, "vecnorm.pkl")

    for fp in tqdm(files, desc=f"{args.symbol}-{args.frame}"):
        train_one_file(
            file_path=fp,
            symbol=args.symbol,
            frame=args.frame,
            args=args,
            agent_dir=agent_dir,
            results_dir=results_dir,
            logs_dir=logs_dir,
            memory_path=memory_path,
            vecnorm_path=vecnorm_path,
            resume_auto=args.resume_auto,
        )

    print("[✅] Training completed.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
