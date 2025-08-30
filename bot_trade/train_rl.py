#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_rl.py — Orchestrator for SB3 PPO with unified config/ modules.
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

from bot_trade.config.rl_paths import dataset_path

# Heavy dependencies (torch, numpy, pandas, stable_baselines3, etc.) are
# imported inside `main` to keep import-time side effects minimal and to
# allow `python -m bot_trade.train_rl --help` to run without them.



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


def auto_shape_resources(args):
    """Auto-derive n_envs/n_steps/batch_size based on hardware when unset."""
    cpu_cores = os.cpu_count() or 2
    gpu = torch.cuda.is_available()
    if getattr(args, "n_envs", 0) <= 0:
        if gpu:
            args.n_envs = 4
        else:
            args.n_envs = min(max(2, cpu_cores // 2), 16)
    if gpu and args.n_envs < 4:
        args.n_envs = 4
    if getattr(args, "batch_size", 0) > args.n_envs * args.n_steps:
        logging.warning("[PPO] batch_size (%d) > n_envs*n_steps (%d) — clipping", args.batch_size, args.n_envs * args.n_steps)
        args.batch_size = args.n_envs * args.n_steps
    clamp_batch(args)
    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        ram_gb = float('nan')
    logging.info("[AUTO] cpu_cores=%s ram_gb=%.1f n_envs=%s n_steps=%s batch_size=%s", cpu_cores, ram_gb, args.n_envs, args.n_steps, args.batch_size)
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


def _spawn_monitor(args):
    """Launch monitor manager in a background process."""
    import subprocess, shutil

    images_out = getattr(args, "monitor_images_out", "").format(
        symbol=args.symbol, frame=args.frame
    )
    root_dir = os.path.abspath(os.path.join(args.results_dir, os.pardir))
    exe = shutil.which("bot-monitor")
    if exe:
        cmd = [exe]
    else:
        cmd = [sys.executable, "-m", "bot_trade.tools.monitor_manager"]
    cmd += [
        "--symbol",
        args.symbol,
        "--frame",
        args.frame,
        "--run-id",
        args.run_id,
        "--refresh",
        str(getattr(args, "monitor_refresh", 10)),
        "--images-out",
        images_out,
        "--base",
        root_dir,
    ]
    try:
        return subprocess.Popen(cmd)
    except Exception as exc:  # pragma: no cover
        logging.warning("monitor launch failed: %s", exc)
        return None



# =============================
# Single-file training job
# =============================

def train_one_file(args, data_file: str) -> bool:
    """Train on a single dataset file with resume capability."""
    mem = load_memory()
    resume_data = None

    if getattr(args, "resume", None):
        rid = args.resume
        if not rid or str(rid).lower() in {"latest", "last", "true", "1"}:
            rid = mem.get("last_run_id")
        if not rid:
            raise SystemExit("[RESUME] no previous run to resume")
        resume_data = resume_from_snapshot(str(rid))
        data_file = resume_data.get("dataset", {}).get("path", data_file)
        run_id = str(rid)
    else:
        run_id = getattr(args, "run_id", None) or new_run_id(args.symbol, args.frame)
    args.run_id = run_id

    data_file = str(dataset_path(data_file))
    logging.info("[DATA] dataset path resolved to %s", data_file)

    # 1) Paths + logging + state
    paths = build_paths(
        args.symbol,
        args.frame,
        agents_dir=args.agents_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
    )
    paths.update(get_paths(args.symbol, args.frame))
    log_queue, listener, _ = create_loggers(paths["results"], args.frame, args.symbol, level=getattr(args, "log_level", logging.INFO))
    ensure_state_files(args.memory_file, args.kb_file)
    logging.info("[PATHS] initialized for %s | %s", args.symbol, args.frame)

    # Pre-load knowledge base if present
    try:
        with open(args.kb_file, "r", encoding="utf-8") as fh:
            kb_data = json.load(fh)
        kb_size = len(kb_data) if isinstance(kb_data, list) else len(kb_data.keys())
        logging.info("[KB] loaded %d entries", kb_size)
    except Exception:
        logging.info("[KB] initialized new knowledge base at %s", args.kb_file)

    cfg = get_config()
    update_manager = UpdateManager(paths, args.symbol, args.frame, cfg)

    # merge CLI overrides into cfg surface
    ppo_cfg = cfg.setdefault("ppo", {})
    ppo_cfg.update({
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "clip_range": float(args.clip_range),
        "n_epochs": int(args.epochs),
    })
    pol_cfg = cfg.setdefault("policy_kwargs", {})
    pol_cfg.update({
        "net_arch": args.policy_kwargs.get("net_arch"),
        "activation_fn": getattr(args.policy_kwargs.get("activation_fn"), "__name__", "relu"),
        "ortho_init": bool(args.policy_kwargs.get("ortho_init", False)),
    })
    eval_cfg = cfg.setdefault("eval", {})
    eval_cfg["enable"] = bool(eval_cfg.get("enable", True))
    eval_cfg["n_episodes"] = int(getattr(args, "eval_episodes", eval_cfg.get("n_episodes", 5)))
    if int(getattr(args, "eval_every_steps", 0)) > 0:
        eval_cfg["eval_freq"] = int(args.eval_every_steps)
    eval_cfg.setdefault("eval_freq", max(int(args.n_steps), 1000))
    eval_cfg.setdefault("save_best", True)
    log_cfg = cfg.setdefault("logging", {})
    log_cfg["step_every"] = int(getattr(args, "log_every_steps", log_cfg.get("step_every", 100)))
    writers = Writers(paths, args.frame, args.symbol)

    # ==== resume state / portfolio ====
    run_state = load_run_state()
    portfolio_cfg = cfg.get("portfolio", {})
    port_state = load_portfolio_state(args.symbol, args.frame)
    if not port_state and portfolio_cfg.get("enable"):
        port_state = reset_with_balance(args.symbol, args.frame, portfolio_cfg.get("balance_start", 0.0))

    mon_proc = _spawn_monitor(args) if getattr(args, "monitor", True) else None
    if mon_proc:
        import atexit
        atexit.register(lambda: mon_proc.terminate())

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

    try:
        import hashlib
        st = os.stat(data_file)
        h = hashlib.sha256()
        with open(data_file, "rb") as fh:
            h.update(fh.read(1024 * 1024))
        dataset_info = {
            "path": data_file,
            "origin": getattr(args, "data_origin", "unknown"),
            "mtime": st.st_mtime,
            "size_bytes": st.st_size,
            "rows": int(getattr(df, "shape", [0])[0]),
            "hash_head": h.hexdigest(),
        }
    except Exception:
        dataset_info = {"path": data_file, "origin": getattr(args, "data_origin", "unknown")}

    cur_cfg = cfg.get("curriculum", {})
    if cur_cfg.get("enable"):
        metric = cur_cfg.get("regime_metric", "volatility")
        series = None
        if metric == "volatility" and "close" in df.columns:
            series = df["close"].pct_change().rolling(30).std()
        elif metric == "atr" and "atr" in df.columns:
            series = pd.to_numeric(df["atr"], errors="coerce")
        if series is not None:
            start_q = float(cur_cfg.get("start_quantile", 0.3))
            thr = series.quantile(start_q)
            df = df.loc[series <= thr]

    # 4) Build env constructors
    env_fns = build_env_fns(
        df,
        frame=args.frame,
        symbol=args.symbol,
        n_envs=int(args.n_envs),
        use_indicators=args.use_indicators,
        config=cfg,
        writers=writers,
        safe=args.safe,
        decisions_jsonl=None,
    )

    # 5) VecEnv (DummyVecEnv if n_envs==1 else SubprocVecEnv)
    vec_env = make_vec_env(
        env_fns,
        n_envs=int(args.n_envs),
        start_method=getattr(args, "mp_start", "spawn"),
        normalize=True,
        seed=getattr(args, "seed", None),
    )

    # expose risk manager for snapshots/resume
    risk_manager = None
    try:
        risk_manager = vec_env.get_attr("risk_engine")[0]
    except Exception:
        pass

    if resume_data:
        env_state = resume_data.get("env_state", {})
        try:
            ptr = env_state.get("ptr_index")
            if ptr is not None:
                vec_env.set_attr("ptr", int(ptr))
        except Exception:
            pass
        try:
            op = env_state.get("open_position", {})
            if op:
                vec_env.set_attr("entry_price", op.get("entry"))
                size = op.get("size")
                side = op.get("side")
                if size is not None and side is not None:
                    amt = float(size) if side == "long" else -float(size)
                    vec_env.set_attr("coin", amt)
        except Exception:
            pass
        try:
            vec_env.set_attr("last_action", env_state.get("last_action"))
            vec_env.set_attr("last_reward", env_state.get("last_reward"))
            vec_env.set_attr("last_reward_components", env_state.get("last_reward_components"))
        except Exception:
            pass
        try:
            import numpy as np, random
            np_state = env_state.get("rng_numpy")
            if np_state is not None:
                np.random.set_state(tuple(np_state))
            py_state = env_state.get("rng_python")
            if py_state is not None:
                random.setstate(tuple(py_state))
        except Exception:
            pass
        if risk_manager:
            rs = resume_data.get("risk_state", {})
            try:
                risk_manager.current_risk = rs.get("risk_pct", risk_manager.current_risk)
            except Exception:
                pass
            try:
                risk_manager.freeze_mode = rs.get("freeze_mode", risk_manager.freeze_mode)
            except Exception:
                pass
            try:
                risk_manager.ema_reward = rs.get("ema_reward", risk_manager.ema_reward)
            except Exception:
                pass
            try:
                risk_manager.max_drawdown = rs.get("max_drawdown", risk_manager.max_drawdown)
            except Exception:
                pass
        try:
            writers.train.last_step = resume_data.get("writers", {}).get("last_artifact_step", 0)
        except Exception:
            pass

    # 6) Try loading VecNormalize running averages
    loaded_vecnorm = False
    if resume_data:
        vp = resume_data.get("global", {}).get("vecnorm_path")
        if vp and os.path.exists(vp):
            try:
                if hasattr(vec_env, "load_running_average"):
                    vec_env.load_running_average(vp)
                    loaded_vecnorm = True
                    logging.info("[VECNORM] loaded running average from snapshot")
            except Exception as e:
                logging.warning("[VECNORM] load failed (snapshot): %s", e)
    if not loaded_vecnorm:
        cand = "vecnorm_best" if getattr(args, "resume_best", False) else "vecnorm_pkl"
        p = paths.get(cand)
        if p and os.path.exists(p):
            try:
                if hasattr(vec_env, "load_running_average"):
                    vec_env.load_running_average(p)
                    loaded_vecnorm = True
                    logging.info("[VECNORM] loaded running average from %s", cand)
            except Exception as e:
                logging.warning("[VECNORM] load failed (%s): %s", cand, e)
    if not loaded_vecnorm:
        logging.info("[VECNORM] no previous running average found.")

    # 7) Action space detection
    action_space, is_discrete = detect_action_space(vec_env)
    logging.info("[ENV] action_space=%s | is_discrete=%s", action_space, is_discrete)

    # 8) Batch clamping
    clamp_batch(args)

    # 8b) Evaluation environment
    eval_env_fns = build_env_fns(
        df,
        frame=args.frame,
        symbol=args.symbol,
        n_envs=1,
        use_indicators=args.use_indicators,
        config=cfg,
        writers=None,
        safe=args.safe,
        decisions_jsonl=None,
    )
    eval_env = make_vec_env(
        eval_env_fns,
        n_envs=1,
        start_method=getattr(args, "mp_start", "spawn"),
        normalize=True,
        seed=getattr(args, "seed", None),
    )

    # 9) Resume or build fresh model
    model = None
    resume_path = None
    if resume_data:
        g = resume_data.get("global", {})
        resume_path = g.get("checkpoint_path") or g.get("best_path")
    elif getattr(args, "resume_best", False):
        if os.path.exists(paths["model_best_zip"]):
            resume_path = paths["model_best_zip"]
        elif os.path.exists(paths["model_zip"]):
            resume_path = paths["model_zip"]
    elif getattr(args, "resume_auto", False):
        if os.path.exists(paths["model_zip"]):
            resume_path = paths["model_zip"]
        elif os.path.exists(paths["model_best_zip"]):
            resume_path = paths["model_best_zip"]
    if resume_path:
        try:
            from stable_baselines3 import PPO
            model = PPO.load(resume_path, env=vec_env, device=args.device_str)
            logging.info("[RESUME] Loaded model from %s", resume_path)
        except Exception as e:
            logging.error("[RESUME] failed to load: %s. Building fresh model.", e)

    if model is None:
        sched = cfg.get("ppo", {}).get("lr_schedule", "constant")
        if sched != "constant" and not callable(args.learning_rate):
            warm = int(cfg["ppo"].get("warmup_steps", 0))
            base_lr = float(args.learning_rate)
            total = int(args.total_steps)
            def lr_schedule(progress):
                step = (1 - progress) * total
                if step < warm:
                    return base_lr * step / max(1, warm)
                pct = (step - warm) / max(1, total - warm)
                return 0.5 * base_lr * (1 + math.cos(math.pi * pct))
            args.learning_rate = lr_schedule
        model = build_ppo(args, vec_env, is_discrete)
        logging.info("[PPO] Built new model (device=%s, use_sde=%s)", args.device_str, bool(args.sde and not is_discrete))

    if resume_data:
        try:
            model.num_timesteps = int(resume_data.get("global", {}).get("global_timesteps", model.num_timesteps))
        except Exception:
            pass

    # 10) Callbacks (base from rl_builders + optional extras)
    base_callbacks = build_callbacks(
        paths, writers, args, update_manager,
        run_id=run_id, risk_manager=risk_manager, dataset_info=dataset_info,
    )
    cb = base_callbacks
    extras = []

    composite_cb = CompositeCallback(update_manager, cfg)
    extras.append(composite_cb)

    eval_cb: Optional[EvalCallback] = None
    if cfg.get("eval", {}).get("enable", True):
        eval_cb = EvalSaveCallback(
            eval_env,
            latest_path=paths["model_zip"],
            best_path=paths["model_best_zip"],
            vecnorm_path=paths["vecnorm_pkl"],
            vecnorm_best=paths["vecnorm_best"],
            best_meta=paths["best_meta"],
            n_eval_episodes=int(cfg["eval"].get("n_episodes", 5)),
            eval_freq=int(cfg["eval"].get("eval_freq", int(args.n_steps))),
            deterministic=True,
            patience=int(cfg["eval"].get("patience_eval_rounds", 3)),
            lr_factor=float(cfg["eval"].get("lr_decay_factor", 0.5)),
            lr_limit=int(cfg["eval"].get("lr_decay_limit", 2)),
        )
        extras.append(eval_cb)

    if BenchmarkCallback is not None:
        extras.append(BenchmarkCallback(frame=args.frame, symbol=args.symbol, writers=writers, every_sec=15))
    if getattr(args, "safe", False) and StrictDataSanityCallback is not None:
        extras.append(StrictDataSanityCallback(writers=writers, raise_on_issue=True))
    if extras:
        if CallbackList is not None and hasattr(base_callbacks, "callbacks"):
            base_callbacks.callbacks.extend(extras)  # type: ignore[attr-defined]
            cb = base_callbacks
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
        summary = {
            "best_model_path": getattr(eval_cb, "best_model_path", None),
            "metric": getattr(eval_cb, "best_mean_reward", None),
        }
        try:
            update_manager.on_eval_end(summary)
        except Exception:
            pass
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
        try:
            model.save(paths["model_zip"])  # final
            if hasattr(vec_env, "save_running_average"):
                vec_env.save_running_average(paths["vecnorm_pkl"])  # last
            logging.info("[SAVE] model -> %s | vecnorm -> %s", paths["model_zip"], paths["vecnorm_pkl"])
        except Exception as e:
            logging.error("[SAVE] failed: %s", e)
        try:
            vec_env.close()
        except Exception:
            pass
        try:
            update_manager.on_training_end(summary)
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass
        logging.shutdown()
        try:
            writers.close()
        except Exception:
            pass

    # final snapshot after training
    try:
        vecnorm = None
        try:
            vecnorm = model.get_vec_normalize_env()
        except Exception:
            pass
        commit_snapshot(run_id, make_snapshot(args, vec_env, model, vecnorm, writers, risk_manager, dataset_info))
    except Exception:
        pass

    if cfg.get("knowledge", {}).get("run_after_training", False):
        try:
            import subprocess
            subprocess.run([
                sys.executable,
                "-m",
                "bot_trade.tools.knowledge_sync",
                "--results",
                paths.get("results", "results"),
                "--agents",
                paths.get("agents", "agents"),
                "--to",
                os.path.join("memory", "knowledge", "summary.json"),
                "--summarize",
                "--propose-config",
            ], check=False)
        except Exception as e:
            logging.warning("[KNOWLEDGE] sync failed: %s", e)

    if cfg.get("reports", {}).get("enable", False):
        try:
            from bot_trade.tools.generate_markdown_report import generate_summary
            generate_summary(paths, args.symbol, args.frame)
        except Exception as e:
            logging.warning("[REPORT] generation failed: %s", e)

    if cfg.get("self_learning", {}).get("enable", False):
        try:
            from bot_trade.ai_core.self_improver import propose_config_updates, apply_updates_to_config
            updates = propose_config_updates()
            if cfg["self_learning"].get("writeback_config", False):
                apply_updates_to_config(updates)
            else:
                logging.info("[SELF_LEARN] proposed updates: %s", list(updates.keys()))
        except Exception as e:
            logging.warning("[SELF_LEARN] failed: %s", e)

    # save run/portfolio state
    try:
        save_run_state({
            "symbol": args.symbol,
            "frame": args.frame,
            "last_step": int(getattr(model, "num_timesteps", 0)),
        })
        if portfolio_cfg.get("enable"):
            save_portfolio_state(args.symbol, args.frame, port_state)
    except Exception:
        pass

    if mon_proc:
        try:
            mon_proc.terminate()
        except Exception:
            pass

    return True


# =============================
# Main (playlist or single job)
# =============================

def main():
    from bot_trade.config.rl_args import parse_args, finalize_args

    args = parse_args()

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        print(
            "[ERROR] PyTorch is required for training.\n"
            "For CPU:   conda install pytorch cpuonly -c pytorch\n"
            "For CUDA:  conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia"
        )
        raise SystemExit(1)

    global torch, np, pd, psutil, subprocess, shutil
    global build_env_fns, make_vec_env, detect_action_space, build_ppo, build_callbacks
    global build_paths, ensure_state_files, get_paths
    global Writers, create_loggers, setup_worker_logging, UpdateManager, CompositeCallback, get_config
    global EvalCallback, EvalSaveCallback, load_run_state, save_run_state, MemoryManager
    global load_memory, commit_snapshot, make_snapshot, resume_from_snapshot, new_run_id
    global load_portfolio_state, save_portfolio_state, reset_with_balance
    global discover_files, read_one, LoadOptions, load_dataset, add_strategy_features, _HAS_READ_ONE
    global CallbackList, BenchmarkCallback, StrictDataSanityCallback

    import math, psutil, numpy as np, pandas as pd, subprocess, shutil

    from bot_trade.config.rl_builders import (
        build_env_fns,
        make_vec_env,
        detect_action_space,
        build_ppo,
        build_callbacks,
    )
    from bot_trade.config.rl_paths import build_paths, ensure_state_files, get_paths
    from bot_trade.config.rl_writers import Writers  # Writers bundle (train/eval/...)
    from bot_trade.config.log_setup import create_loggers, setup_worker_logging
    from bot_trade.config.update_manager import UpdateManager
    from bot_trade.config.rl_callbacks import CompositeCallback
    from bot_trade.config.env_config import get_config
    from stable_baselines3.common.callbacks import EvalCallback
    from bot_trade.tools.run_state import load_state as load_run_state, save_state as save_run_state
    from bot_trade.tools.memory_manager import (
        MemoryManager,
        load_memory,
        commit_snapshot,
        make_snapshot,
        resume_from_snapshot,
    )
    from bot_trade.tools.runctx import new_run_id
    from bot_trade.ai_core.portfolio import (
        load_state as load_portfolio_state,
        save_state as save_portfolio_state,
        reset_with_balance,
    )
    try:
        from bot_trade.config.loader import discover_files, read_one, LoadOptions  # preferred
        _HAS_READ_ONE = True
    except Exception:
        from bot_trade.config.loader import discover_files, load_dataset  # fallback signature
        _HAS_READ_ONE = False
    try:
        from bot_trade.config.strategy_features import add_strategy_features
    except Exception:
        add_strategy_features = None  # pragma: no cover
    try:
        from stable_baselines3.common.callbacks import CallbackList
        from bot_trade.config.rl_callbacks import BenchmarkCallback, StrictDataSanityCallback
    except Exception:  # pragma: no cover
        BenchmarkCallback = None
        StrictDataSanityCallback = None
        CallbackList = None

    class EvalSaveCallback(EvalCallback):
        """EvalCallback that maintains latest and best checkpoints."""

        def __init__(
            self,
            eval_env,
            latest_path: str,
            best_path: str,
            vecnorm_path: str,
            vecnorm_best: str,
            best_meta: str,
            patience: int = 3,
            lr_factor: float = 0.5,
            lr_limit: int = 2,
            **kwargs,
        ):
            save_dir = os.path.dirname(best_path)
            os.makedirs(save_dir, exist_ok=True)
            super().__init__(eval_env, best_model_save_path=save_dir, **kwargs)
            self._latest_path = latest_path
            self._best_path = best_path
            self._vecnorm_path = vecnorm_path
            self._vecnorm_best = vecnorm_best
            self._best_meta = best_meta
            self._last_mtime = 0.0
            self._patience = int(patience)
            self._lr_factor = float(lr_factor)
            self._lr_limit = int(lr_limit)
            self._no_improve = 0
            self._lr_updates = 0
            self._best_metric = -float("inf")

        def _save_vecnorm(self, path: str) -> None:
            try:
                vec = self.model.get_vec_normalize_env()
                if vec is None:
                    return
                if hasattr(vec, "save_running_average"):
                    vec.save_running_average(path)
                elif hasattr(vec, "save"):
                    vec.save(path)
            except Exception:
                pass

        def _on_step(self) -> bool:  # type: ignore[override]
            result = super()._on_step()
            if self.eval_freq and self.n_calls % self.eval_freq == 0:
                try:
                    self.model.save(self._latest_path)
                    self._save_vecnorm(self._vecnorm_path)
                except Exception:
                    pass
                src = getattr(self, "best_model_path", None)
                if src and os.path.exists(src):
                    try:
                        mtime = os.path.getmtime(src)
                        if mtime > self._last_mtime:
                            shutil.copy(src, self._best_path)
                            shutil.copy(self._vecnorm_path, self._vecnorm_best)
                            meta = {
                                "metric": float(getattr(self, "best_mean_reward", 0.0)),
                                "timestamp": dt.datetime.utcnow().isoformat(),
                            }
                            with open(self._best_meta, "w", encoding="utf-8") as fh:
                                json.dump(meta, fh)
                            self._last_mtime = mtime
                    except Exception:
                        pass
                cur = getattr(self, "last_mean_reward", None)
                if cur is not None:
                    if cur > self._best_metric:
                        self._best_metric = cur
                        self._no_improve = 0
                    else:
                        self._no_improve += 1
                        if self._no_improve >= self._patience:
                            if self._lr_updates < self._lr_limit:
                                try:
                                    for g in self.model.policy.optimizer.param_groups:
                                        g["lr"] *= self._lr_factor
                                    self._lr_updates += 1
                                    logging.warning(
                                        "[Eval] LR reduced to %s",
                                        self.model.policy.optimizer.param_groups[0]["lr"],
                                    )
                                except Exception:
                                    pass
                                self._no_improve = 0
                            else:
                                logging.warning("[Eval] no-improvement")
            return result

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    args = auto_shape_resources(args)
    args = validate_args(args)
    args = finalize_args(args, is_continuous=None)  # allow builders to decide SDE later

    # Resolve dataset root (CLI > config.yaml > default)
    cfg = get_config()
    cfg_paths = cfg.get("project", {}).get("paths", {}) if isinstance(cfg, dict) else {}
    cfg_root = cfg_paths.get("ready_dir") or cfg_paths.get("data_dir")
    cli_root = getattr(args, "data_root", None)
    data_root = cli_root or cfg_root or "data"
    origin = "cli" if cli_root else ("config" if cfg_root else "default")
    data_root = str(dataset_path(data_root))
    args.data_root = data_root
    args.data_origin = origin
    logging.info("[DATA] root resolved to %s (origin=%s)", data_root, origin)

    mm = MemoryManager()
    mm.log_event("start", {"args": vars(args)})

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
            from bot_trade.config.rl_args import parse_net_arch
            if isinstance(getattr(job_args, "net_arch", None), str):
                job_args.policy_kwargs = parse_net_arch(job_args.net_arch)
            files = discover_files(job_args.frame, job_args.symbol)
            for fp in files:
                train_one_file(job_args, fp)
        return

    # Single-job: pick first matching file
    files = discover_files(args.frame, args.symbol, data_root=args.data_root)
    if not files:
        raise FileNotFoundError(
            f"No data files for {args.symbol}-{args.frame} in {args.data_root}"
        )
    data_file = files[0]
    logging.info("[DATA] using file %s", data_file)
    train_one_file(args, data_file)
    mm.snapshot({"data_file": data_file, "origin": args.data_origin})
    mm.log_event("end", {"status": "ok"})


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
