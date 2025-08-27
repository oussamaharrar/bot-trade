#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train_RL.py — Orchestrator for SB3 PPO with unified config/ modules.
- Central logging via config.log_setup
- Paths/state via config.rl_paths
- Data via config.loader (no double indicators)
- Env via config.env_trading (Gymnasium)
- Writers bundle via config.rl_writers
- Callbacks via config.rl_builders (+ optional extras)

Notes:
* Respects --quiet-device-report from rl_args.
* Uses args.mp_start for SubprocVecEnv start method.
* Loads VecNormalize running averages if present.
* Disables gSDE for Discrete action spaces automatically.
"""

from __future__ import annotations
import os, sys, time, json, logging, datetime as dt
from typing import Any, Dict, Optional

import numpy as np
import torch

# === Args/Builders/Paths/Logging/Data ===
from config.rl_args import parse_args, finalize_args
from config.rl_builders import (
    build_env_fns,
    make_vec_env,
    detect_action_space,
    build_ppo,
    build_callbacks,
)
from config.rl_paths import build_paths, ensure_state_files, get_paths
from config.rl_writers import Writers  # Writers bundle (train/eval/...)
from config.update_manager import UpdateManager
from config.rl_callbacks import CompositeCallback
from config.log_setup import create_loggers, setup_worker_logging

# Loader (support both single-file and discover-based flows)
try:
    from config.loader import discover_files, read_one, LoadOptions  # preferred
    _HAS_READ_ONE = True
except Exception:
    from config.loader import discover_files, load_dataset  # fallback signature
    _HAS_READ_ONE = False

# (Optional) strategy_features for pre-computing indicators if loader supports it
try:
    from config.strategy_features import add_strategy_features
except Exception:
    add_strategy_features = None  # pragma: no cover

# Optional extras
try:
    from stable_baselines3.common.callbacks import CallbackList
    from config.rl_callbacks import BenchmarkCallback, StrictDataSanityCallback
except Exception:  # pragma: no cover
    BenchmarkCallback = None
    StrictDataSanityCallback = None
    CallbackList = None


# =============================
# Validation / helpers
# =============================

def validate_args(args):
    """Light validation + device string."""
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        try:
            d = int(args.device)
            if d < 0 or d >= torch.cuda.device_count():
                logging.warning("[ARGS] device index %s خارج النطاق؛ سيتم استخدام 0", args.device)
                args.device = 0
        except Exception:
            args.device = 0
        args.device_str = f"cuda:{int(args.device)}"
    else:
        args.device_str = "cpu"

    for k in ("n_envs", "n_steps", "batch_size", "total_steps"):
        if getattr(args, k, 0) <= 0:
            raise ValueError(f"[ARGS] {k} يجب أن يكون > 0.")
    return args


def clamp_batch(args):
    """Ensure batch_size respects n_envs*n_steps and is divisible by n_envs."""
    rollout = int(args.n_envs) * int(args.n_steps)
    if int(args.batch_size) > rollout:
        logging.warning("[PPO] batch_size (%d) > n_envs*n_steps (%d) — سيتم تقليمه", args.batch_size, rollout)
    batch = min(int(args.batch_size), rollout)
    batch = (batch // int(args.n_envs)) * int(args.n_envs)
    args.batch_size = max(int(args.n_envs), int(batch))
    return args


def _maybe_print_device_report(args):
    if getattr(args, "quiet_device_report", False):
        return
    print("========== DEVICE REPORT ==========")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"CUDA device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        try:
            idx = max(0, int(args.device))
            torch.cuda.set_device(idx)
            print(f"Selected device index: {idx} -> {torch.cuda.get_device_name(idx)}")
        except Exception:
            pass
    try:
        print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    except Exception:
        pass
    print("===================================")


# =============================
# Single-file training job
# =============================

def train_one_file(args, data_file: str) -> bool:
    # 1) Paths + logging + state
    paths = build_paths(
        args.symbol,
        args.frame,
        agents_dir=args.agents_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
    )
    log_queue, listener, _ = create_loggers(paths["results"], args.frame, args.symbol)
    ensure_state_files(args.memory_file, args.kb_file)
    logging.info("[PATHS] initialized for %s | %s", args.symbol, args.frame)

    writers = Writers(paths, args.frame, args.symbol)
    # Update manager coordinates all disk writes
    um_paths = get_paths(args.symbol, args.frame)
    update_manager = UpdateManager(um_paths, args.symbol, args.frame, cfg=None)

    # 2) Device report (optional)
    _maybe_print_device_report(args)

    # 3) Load data once (for single-job training)
    if _HAS_READ_ONE:
        df = read_one(data_file, opts=LoadOptions(numeric_only=True, add_features=False, safe=args.safe))
    else:
        df = load_dataset(
            data_file,
            symbol=args.symbol,
            frame=args.frame,
            use_indicators=args.use_indicators,
            add_features_fn=add_strategy_features,
            safe=args.safe,
        )

    # 4) Build env constructors
    env_fns = build_env_fns(
        df,
        frame=args.frame,
        symbol=args.symbol,
        n_envs=int(args.n_envs),
        use_indicators=args.use_indicators,
        config=None,
        writers=writers,
        safe=args.safe,
        decisions_jsonl=paths.get("decisions_jsonl"),
    )

    # 5) VecEnv (DummyVecEnv if n_envs==1 else SubprocVecEnv)
    vec_env = make_vec_env(
        env_fns,
        n_envs=int(args.n_envs),
        start_method=getattr(args, "mp_start", "spawn"),
        normalize=True,
        seed=getattr(args, "seed", None),
    )

    # 6) Try loading VecNormalize running averages (prefer best)
    loaded_vecnorm = False
    for cand in ("vecnorm_best", "vecnorm_pkl"):
        p = paths.get(cand)
        if not p:
            continue
        try:
            if os.path.exists(p):
                if hasattr(vec_env, "load_running_average"):
                    vec_env.load_running_average(p)
                    loaded_vecnorm = True
                    logging.info("[VECNORM] loaded running average from %s", cand)
                    break
        except Exception as e:
            logging.warning("[VECNORM] load failed (%s): %s", cand, e)
    if not loaded_vecnorm:
        logging.info("[VECNORM] no previous running average found.")

    # 7) Action space detection
    action_space, is_discrete = detect_action_space(vec_env)
    logging.info("[ENV] action_space=%s | is_discrete=%s", action_space, is_discrete)

    # 8) Batch clamping
    clamp_batch(args)

    # 9) Resume or build fresh model
    model = None
    resume_path = None
    if args.resume_auto:
        if os.path.exists(paths["model_best_zip"]):
            resume_path = paths["model_best_zip"]
        elif os.path.exists(paths["model_zip"]):
            resume_path = paths["model_zip"]
    if resume_path:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(resume_path, env=vec_env, device=args.device_str)
            logging.info("[RESUME] Loaded model from %s", resume_path)
        except Exception as e:
            logging.error("[RESUME] failed to load: %s. Building fresh model.", e)

    if model is None:
        model = build_ppo(args, vec_env, is_discrete)
        logging.info("[PPO] Built new model (device=%s, use_sde=%s)", args.device_str, bool(args.sde and not is_discrete))

    # 10) Callbacks (base from rl_builders + optional extras)
    base_callbacks = build_callbacks(paths, writers, args)
    cb = base_callbacks
    extras = [CompositeCallback(update_manager, log_every=100)]
    if BenchmarkCallback is not None:
        extras.append(BenchmarkCallback(frame=args.frame, symbol=args.symbol, writers=writers, every_sec=15))
    if getattr(args, "safe", False) and StrictDataSanityCallback is not None:
        extras.append(StrictDataSanityCallback(writers=writers, raise_on_issue=True))
    if extras:
        if CallbackList is not None and hasattr(base_callbacks, "callbacks"):
            base_callbacks.callbacks.extend(extras)  # type: ignore[attr-defined]
        else:
            cb = CallbackList([base_callbacks] + extras)  # type: ignore[arg-type]

    # 11) Learn
    status = "finished"
    try:
        logging.info("[🚀] Training for total_steps=%s .", args.total_steps)
        model.learn(total_timesteps=int(args.total_steps), callback=cb, progress_bar=args.progress)
    except KeyboardInterrupt:
        status = "interrupted"
        logging.warning("[INTERRUPT] KeyboardInterrupt — saving.")
    except Exception as e:
        status = f"error:{e.__class__.__name__}"
        logging.exception("[ERROR] during learn: %s", e)
        raise
    finally:
        # Save model and VecNormalize
        try:
            model.save(paths["model_zip"])  # final
            if hasattr(vec_env, "save_running_average"):
                vec_env.save_running_average(paths["vecnorm_pkl"])  # last
            logging.info("[SAVE] model -> %s | vecnorm -> %s", paths["model_zip"], paths["vecnorm_pkl"])
        except Exception as e:
            logging.error("[SAVE] failed: %s", e)
        # Stop QueueListener safely
        try:
            listener.stop()
        except Exception:
            pass
        # Close env and writers
        try:
            vec_env.close()
        except Exception:
            pass
        try:
            writers.close()
        except Exception:
            pass

    # Train/Eval summary
    try:
        now = dt.datetime.utcnow().isoformat()
        file_name = os.path.basename(data_file)
        end_ts = int(getattr(model, "num_timesteps", 0))
        writers.train.write([now, args.frame, args.symbol, file_name, end_ts, status])
        ep = getattr(model, "ep_info_buffer", None)
        if ep and len(ep) > 0:
            avg = float(np.mean([x.get("r", 0.0) for x in ep]))
            writers.eval.write([now, args.frame, args.symbol, "ep_rew_mean", avg])
    except Exception:
        pass

    return True


# =============================
# Main (playlist or single job)
# =============================

def main():
    args = parse_args()
    args = validate_args(args)
    args = finalize_args(args, is_continuous=None)  # allow builders to decide SDE later

    # Ensure resume-auto can accept optional value
    if not hasattr(args, "resume_auto"):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--resume-auto",
            nargs="?",
            const="latest",
            default=None,
            help="Resume from last checkpoint (optionally specify 'latest')."
        )

    # Playlist mode — sequential in one process
    if args.playlist and os.path.exists(args.playlist):
        import yaml
        with open(args.playlist, "r", encoding="utf-8") as f:
            jobs = yaml.safe_load(f) or []
        for j in jobs:
            class _NS: ...
            job_args = _NS()
            for k, v in vars(args).items():
                setattr(job_args, k, v)
            for k, v in (j or {}).items():
                setattr(job_args, k.replace("-", "_"), v)
            job_args = validate_args(job_args)
            # optional: refresh policy kwargs if net_arch string provided
            from config.rl_args import parse_net_arch
            if isinstance(getattr(job_args, "net_arch", None), str):
                job_args.policy_kwargs = parse_net_arch(job_args.net_arch)
            files = discover_files(job_args.frame, job_args.symbol)
            for fp in files:
                train_one_file(job_args, fp)
        return

    # Single-job: pick first matching file
    files = discover_files(args.frame, args.symbol)
    if not files:
        raise FileNotFoundError(f"No data files for {args.symbol}-{args.frame}")
    data_file = files[0]
    train_one_file(args, data_file)


if __name__ == "__main__":
    main()
