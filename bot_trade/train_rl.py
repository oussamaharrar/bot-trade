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
import os, sys, time, json, logging, datetime as dt, math, threading, pathlib, zipfile
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

from bot_trade.config.rl_paths import (
    dataset_path,
    RunPaths,
    ensure_contract,
    DEFAULT_KB_FILE,
)
from bot_trade.config.device import normalize_device, maybe_print_device_report
from bot_trade.config.rl_callbacks import _save_vecnorm
from bot_trade.config.rl_args import validate_args, clamp_batch, auto_shape_resources
from bot_trade.config.log_setup import drain_log_queue
from bot_trade.tools import export_charts
from bot_trade.tools.evaluate_model import evaluate_for_run
from bot_trade.tools.eval_run import evaluate_run
from bot_trade.tools.kb_writer import kb_append
from bot_trade.tools.monitor_launch import spawn_monitor_manager
from bot_trade.tools.run_state import (
    update_portfolio_state,
    write_run_state_files,
)
from bot_trade.tools._headless import ensure_headless_once

# Heavy dependencies (torch, numpy, pandas, stable_baselines3, etc.) are
# imported inside `main` to keep import-time side effects minimal and to
# allow `python -m bot_trade.train_rl --help` to run without them.













def _postrun_summary(paths, meta):
    logger = logging.getLogger()
    run_id = meta.get("run_id") or (
        paths.get("run_id") if isinstance(paths, dict) else getattr(paths, "run_id", "<unknown>")
    )
    sym = getattr(paths, "symbol", meta.get("symbol", "?"))
    frm = getattr(paths, "frame", meta.get("frame", "?"))

    algo = (
        getattr(paths, "algo", None)
        if isinstance(paths, RunPaths)
        else (meta.get("algorithm") or "PPO")
    )
    rp = (
        paths
        if isinstance(paths, RunPaths)
        else RunPaths(
            sym,
            frm,
            str(run_id),
            algo,
            kb_file=(paths.get("kb_file") if isinstance(paths, dict) else None),
        )
    )

    eval_summary: dict[str, Any] = meta.get("synthetic_eval") or {}
    if not eval_summary:
        try:
            eval_summary = evaluate_for_run(rp, episodes=int(meta.get("eval_episodes", 3)))
            logger.info("[EVAL] done episodes=%s", meta.get("eval_episodes", 3))
        except Exception as e:
            logger.warning("[EVAL] failed err=%s", e)
            eval_summary = {}

    rp.charts_dir.mkdir(parents=True, exist_ok=True)
    try:
        charts_dir, img_count, row_counts = export_charts.export_run_charts(rp, rp.run_id)
    except Exception:
        try:
            charts_dir, img_count, row_counts = export_charts.export_run_charts(rp, rp.run_id)
        except Exception as e:
            charts_dir = rp.charts_dir
            img_count = 0
            row_counts = {}
            logger.warning("[POSTRUN_EXPORT] export_failed err=%s", e)
    rows_reward = row_counts.get("reward", 0)
    rows_step = row_counts.get("step", 0)
    rows_train = row_counts.get("train", 0)
    rows_risk = row_counts.get("risk", 0)
    rows_callbacks = row_counts.get("callbacks", 0)
    rows_signals = row_counts.get("signals", 0)
    print(
        "[DEBUG_EXPORT] reward_rows=%d step_rows=%d train_rows=%d risk_rows=%d callbacks_rows=%d signals_rows=%d"
        % (
            rows_reward,
            rows_step,
            rows_train,
            rows_risk,
            rows_callbacks,
            rows_signals,
        )
    )
    print(f"[CHARTS] dir={charts_dir.resolve()} images={img_count}")

    agents_base = Path(rp.agents)
    best = agents_base / "deep_rl_best.zip"
    last = agents_base / "deep_rl_last.zip"
    vecnorm = rp.vecnorm
    reward_lines = rows_reward
    callbacks_lines = rows_callbacks
    vec_applied = bool(meta.get("vecnorm_applied", False))
    vec_snapshot = vecnorm.exists()
    best_ok = best.exists()
    last_ok = last.exists()

    steps_this_run = int(meta.get("total_steps", 0))
    try:
        update_portfolio_state(rp.performance_dir / "portfolio_state.json", steps_this_run)
    except Exception as e:
        logger.warning("[PORTFOLIO] update_failed err=%s", e)
    try:
        write_run_state_files(rp.performance_dir, str(run_id))
    except Exception:
        pass

    try:
        from bot_trade.ai_core.portfolio import load_state as load_portfolio_state

        port_state = load_portfolio_state(sym, frm) or {}
    except Exception:
        port_state = {}

    eval_entry = {
        "win_rate": eval_summary.get("win_rate"),
        "sharpe": eval_summary.get("sharpe"),
        "max_drawdown": eval_summary.get("max_drawdown"),
        "avg_trade_pnl": eval_summary.get("avg_trade_pnl"),
    }

    portfolio_entry = {
        "equity": float(port_state.get("equity", 0.0)),
        "cash": float(port_state.get("balance", 0.0)),
        "positions": len(port_state.get("positions", []) or []),
        "step": int(port_state.get("last_update_step", 0)),
    }

    try:
        kb_entry = {
            "run_id": run_id,
            "symbol": sym,
            "frame": frm,
            "ts": dt.datetime.utcnow().isoformat(),
            "images": img_count,
            "rows_reward": rows_reward,
            "rows_step": rows_step,
            "rows_train": rows_train,
            "rows_risk": rows_risk,
            "rows_callbacks": rows_callbacks,
            "rows_signals": rows_signals,
            "vecnorm_applied": vec_applied,
            "vecnorm_snapshot_saved": vec_snapshot,
            "best": best_ok,
            "last": last_ok,
            "best_model_path": str(rp.best_model.resolve()),
            "eval": eval_entry,
            "portfolio": portfolio_entry,
            "notes": str(meta.get("notes", "")),
        }
        kb_append(rp, kb_entry)
    except Exception as e:
        logger.warning("[KB] append_failed err=%s", e)

    # legacy global run_state writes removed; perf_dir now authoritative

    win_rate = eval_summary.get("win_rate")
    sharpe = eval_summary.get("sharpe")
    line = (
        f"[POSTRUN] run_id={run_id} symbol={sym} frame={frm} "
        f"charts={charts_dir.resolve()} images={img_count} reward_lines={reward_lines} "
        f"step_lines={rows_step} train_lines={rows_train} risk_lines={rows_risk} signals_lines={rows_signals} "
        f"vecnorm_applied={str(vec_applied).lower()} vecnorm_snapshot={str(vec_snapshot).lower()} "
        f"best={str(best_ok).lower()} last={str(last_ok).lower()} "
        f"eval_win_rate={(f'{win_rate:.3f}' if win_rate is not None else 'null')} "
        f"eval_sharpe={(f'{sharpe:.3f}' if sharpe is not None else 'null')}"
    )
    logger.info(line)
    print(line, flush=True)
    return {
        "images": img_count,
        "rows_reward": reward_lines,
        "rows_step": rows_step,
        "rows_train": rows_train,
        "rows_risk": rows_risk,
        "rows_callbacks": callbacks_lines,
        "rows_signals": rows_signals,
        "best": best_ok,
        "last": last_ok,
        "vecnorm_applied": vec_applied,
        "vecnorm_snapshot_saved": vec_snapshot,
        "eval_summary": eval_summary,
    }


def _try_apply_vecnorm(venv, cfg, paths, logger):
    from stable_baselines3.common.vec_env import VecNormalize
    get = paths.get if isinstance(paths, dict) else lambda k: getattr(paths, k, None)
    candidates = [
        pathlib.Path(p) for p in (get("vecnorm_best"), get("vecnorm_last"), get("vecnorm")) if p
    ]
    chosen = next((p for p in candidates if p.exists()), None)
    if cfg.vecnorm and chosen:
        venv = VecNormalize.load(str(chosen), venv)
        venv.training = True
        venv.norm_obs = cfg.norm_obs
        venv.norm_reward = cfg.norm_reward
        logger.info("[VECNORM] applied=True path=%s", chosen)
        return venv, True
    logger.info("[VECNORM] applied=False")
    return venv, False

def _manage_models(paths: Dict[str, str], summary: Dict[str, Any], run_id: str) -> str | None:
    """Handle best/archive model rotation.

    Returns the absolute path to the current best model if available.
    """
    import json, shutil, csv, datetime as dt
    from bot_trade.tools.memory_manager import atomic_json, load_memory
    from bot_trade.config.rl_paths import memory_dir, _atomic_replace, stamp_name

    best_model = pathlib.Path(paths["best_model"])
    last_model = pathlib.Path(paths["last_model"])
    archive_dir = pathlib.Path(paths["archive_dir"])
    archive_best_dir = pathlib.Path(paths["archive_best_dir"])
    vecnorm = pathlib.Path(paths["vecnorm"])
    vecnorm_best = pathlib.Path(paths["vecnorm_best"])
    best_meta = pathlib.Path(paths["best_meta"])

    for d in (archive_dir, archive_best_dir):
        d.mkdir(parents=True, exist_ok=True)

    metric_new = summary.get("metric")
    candidate = pathlib.Path(summary.get("best_model_path") or last_model)
    ts = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    best_metric = float("-inf")
    if best_meta.exists():
        try:
            data = json.loads(best_meta.read_text(encoding="utf-8"))
            metric_obj = data.get("metric")
            if isinstance(metric_obj, dict):
                best_metric = float(metric_obj.get("value", float("-inf")))
            else:
                best_metric = float(metric_obj)
        except Exception:
            pass

    best_path: Path | None = None
    if metric_new is not None and candidate.exists() and (
        metric_new >= best_metric or not best_model.exists()
    ):
        archived_prev: Path | None = None
        if best_model.exists():
            archived_prev = archive_best_dir / stamp_name(
                best_model.stem, run_id, ts, best_model.suffix
            )
            best_model.replace(archived_prev)
            vec_arch: Path | None = None
            if vecnorm_best.exists():
                vec_arch = archive_best_dir / stamp_name(
                    vecnorm_best.stem, run_id, ts, vecnorm_best.suffix
                )
                shutil.copy2(vecnorm_best, vec_arch)
            index = archive_best_dir / "index.csv"
            exists = index.exists()
            with open(index, "a", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                if not exists:
                    w.writerow(["ts", "run_id", "metric", "model_path", "vecnorm_path"])
                w.writerow([ts, run_id, best_metric, str(archived_prev), str(vec_arch) if vec_arch else ""])
        _atomic_replace(candidate, best_model)
        if vecnorm.exists():
            _atomic_replace(vecnorm, vecnorm_best)
        meta = {
            "run_id": run_id,
            "ts": dt.datetime.utcnow().isoformat(),
            "metric": {"name": "metric", "value": float(metric_new)},
            "model": str(best_model.resolve()),
            "vecnorm": str(vecnorm_best.resolve()) if vecnorm_best.exists() else None,
        }
        with open(best_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        logging.info("[BEST] promoted -> %s; archived_prev_best -> %s", best_model, archived_prev)
        best_path = best_model
    else:
        if candidate.exists():
            archived = archive_dir / stamp_name(
                candidate.stem, run_id, ts, candidate.suffix
            )
            shutil.copy2(candidate, archived)
            logging.info("[ARCHIVE] stored -> %s", archived)
        if best_model.exists():
            best_path = best_model

    try:
        mem = load_memory()
        run_mem = mem.setdefault("runs", {}).setdefault(run_id, {})
        run_mem["best_model_path"] = str(best_path) if best_path else None
        run_mem["best_model"] = str(best_path) if best_path else None
        atomic_json(memory_dir() / "memory.json", mem)
    except Exception:
        pass

    return str(best_path) if best_path else None



# =============================
# Single-file training job
# =============================

def train_one_file(args, data_file: str) -> bool:
    """Train on a single dataset file with resume capability."""
    mem = load_memory()
    resume_data = None
    mon_proc = None

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
    run_meta = {
        "run_id": run_id,
        "symbol": args.symbol,
        "frame": args.frame,
        "algorithm": getattr(args, "algorithm", "PPO"),
        "total_steps": int(getattr(args, "total_steps", 0)),
        "eval_episodes": int(getattr(args, "eval_episodes", 3)),
        "export_min_images": int(getattr(args, "export_min_images", 5)),
        "debug_export": bool(getattr(args, "debug_export", False)),
    }

    data_file = str(dataset_path(data_file))
    logging.info("[DATA] dataset path resolved to %s", data_file)
    assert hasattr(pathlib, "Path"), "[PATH] pathlib.Path missing"
    # do NOT reassign `Path` anywhere in this function
    if not pathlib.Path(data_file).exists():
        print(f"[DATA] missing dataset: {data_file}", file=sys.stderr)
        sys.exit(3)

    cfg = get_config()
    algo = (
        getattr(args, "algorithm", None)
        or cfg.get("rl", {}).get("algorithm", "PPO")
    ).upper()
    args.algorithm = algo
    logging.info("[ALG] Using %s", algo)

    # 1) Paths + logging + state
    paths_obj = RunPaths(args.symbol, args.frame, run_id, algo, kb_file=args.kb_file)
    paths_obj.ensure()
    paths = paths_obj.as_dict()
    ensure_contract(paths)
    log_queue, listener, _ = create_loggers(
        paths["results"],
        args.frame,
        args.symbol,
        level=getattr(args, "log_level", logging.INFO),
        logs_path=paths["logs"],
    )
    try:
        listener.stop()
    except Exception:
        pass
    log_thread = threading.Thread(
        target=drain_log_queue, args=(log_queue, listener), daemon=True
    )
    log_thread.start()
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

    update_manager = UpdateManager(paths, args.symbol, args.frame, run_id, cfg)

    cfg.setdefault("rl", {})["algorithm"] = algo

    # merge CLI overrides into cfg surface
    if algo == "SAC":
        sac_cfg = cfg.setdefault("sac", {})
        if args.buffer_size is not None:
            sac_cfg["buffer_size"] = int(args.buffer_size)
        if args.learning_starts is not None:
            sac_cfg["learning_starts"] = int(args.learning_starts)
        if args.train_freq is not None:
            sac_cfg["train_freq"] = int(args.train_freq)
        if args.gradient_steps is not None:
            sac_cfg["gradient_steps"] = int(args.gradient_steps)
        if args.batch_size is not None:
            sac_cfg["batch_size"] = int(args.batch_size)
        if args.tau is not None:
            sac_cfg["tau"] = float(args.tau)
        if args.ent_coef is not None:
            sac_cfg["ent_coef"] = args.ent_coef
        if getattr(args, "sac_gamma", None) is not None:
            sac_cfg["gamma"] = float(args.sac_gamma)
    else:
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
    writers = Writers(paths, run_id)

    # ==== resume state / portfolio ====
    run_state = load_run_state()
    portfolio_cfg = cfg.get("portfolio", {})
    port_state = load_portfolio_state(args.symbol, args.frame)
    if not port_state and portfolio_cfg.get("enable"):
        port_state = reset_with_balance(args.symbol, args.frame, portfolio_cfg.get("balance_start", 0.0))

    # 2) Device report (optional)
    maybe_print_device_report(args)

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
    vecnorm_ref = vec_env if getattr(args, "vecnorm", False) else None

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

    # 6) Load VecNormalize statistics
    if getattr(args, "resume_auto", False):
        try:
            vec_path = paths["vecnorm"] if isinstance(paths, dict) else getattr(paths, "vecnorm", None)
            if vec_path and pathlib.Path(vec_path).exists():
                args.vecnorm = True
        except Exception:
            pass
    vec_env, vecnorm_applied = _try_apply_vecnorm(vec_env, args, paths, logging)
    vecnorm_ref = vec_env if getattr(args, "vecnorm", False) else None
    run_meta["vecnorm_applied"] = vecnorm_applied
    paths_obj.write_run_meta({"vecnorm_applied": vecnorm_applied})

    # 7) Action space detection
    action_space, is_discrete = detect_action_space(vec_env)
    logging.info("[ENV] action_space=%s | is_discrete=%s", action_space, is_discrete)

    # 8) Batch clamping
    n_envsn_steps = int(args.n_envs) * int(args.n_steps)
    if int(args.batch_size) > n_envsn_steps:
        logging.warning(
            "[PPO] batch_size (%d) > n_envsn_steps (%d) — clipping",
            args.batch_size,
            n_envsn_steps,
        )
        args.batch_size = n_envsn_steps
    logging.info("[PPO] effective batch_size=%d", args.batch_size)

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
        normalize=False,
        seed=getattr(args, "seed", None),
    )
    vecnorm_path = Path(paths.get("vecnorm_pkl", paths.get("vecnorm", "")))
    vecnorm_snapshot_exists = vecnorm_path.exists()
    if getattr(args, "vecnorm", False) or vecnorm_snapshot_exists:
        from stable_baselines3.common.vec_env import VecNormalize

        if vecnorm_snapshot_exists:
            eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        else:
            eval_env = VecNormalize(
                eval_env,
                training=False,
                norm_obs=args.norm_obs,
                norm_reward=args.norm_reward,
            )
        eval_env.training = False
        eval_env.norm_obs = args.norm_obs
        eval_env.norm_reward = args.norm_reward

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
        from stable_baselines3 import PPO as _PPO, SAC as _SAC

        Loader = _SAC if algo == "SAC" else _PPO
        best_path = Path(paths.get("model_best_zip", ""))
        try:
            model = Loader.load(
                resume_path, env=vec_env, device=args.device_str, print_system_info=False
            )
            logging.info("[RESUME] Loaded model from %s", resume_path)
        except (zipfile.BadZipFile, ValueError) as e:
            logging.error(
                "[RESUME] bad checkpoint at %s: %s — falling back to fresh model",
                resume_path,
                e,
            )
            if best_path.exists() and Path(resume_path) != best_path:
                try:
                    model = Loader.load(
                        str(best_path),
                        env=vec_env,
                        device=args.device_str,
                        print_system_info=False,
                    )
                    logging.info("[RESUME] loaded BEST instead: %s", best_path)
                except Exception:
                    logging.warning("[RESUME] best load failed; building fresh model")
                    model = None
            else:
                model = None
        except Exception as e:
            logging.error("[RESUME] failed to load: %s. Building fresh model.", e)
            model = None

    if model is None:
        if algo == "SAC":
            model = build_sac(args, vec_env, is_discrete)
            logging.info("[SAC] Built new model (device=%s)", args.device_str)
            if getattr(args, "warmstart_from_ppo", None):
                try:
                    from stable_baselines3 import PPO as _PPO

                    if os.path.exists(args.warmstart_from_ppo):
                        _ppo = _PPO.load(args.warmstart_from_ppo, device=args.device_str)
                        model.policy.features_extractor.load_state_dict(
                            _ppo.policy.features_extractor.state_dict(), strict=False
                        )
                        logger = logging.getLogger(__name__)
                        logger.info(
                            "Warm-started SAC feature extractor from PPO checkpoint."
                        )
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Warm-start skipped: {e}")
        else:
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
            logging.info(
                "[PPO] Built new model (device=%s, use_sde=%s)",
                args.device_str,
                bool(args.sde and not is_discrete),
            )

    if resume_data:
        try:
            model.num_timesteps = int(resume_data.get("global", {}).get("global_timesteps", model.num_timesteps))
        except Exception:
            pass

    # 10) Callbacks (base from rl_builders + optional extras)
    base_callbacks = build_callbacks(
        paths,
        writers,
        args,
        update_manager,
        run_id=run_id,
        risk_manager=risk_manager,
        dataset_info=dataset_info,
        vecnorm_ref=vecnorm_ref,
    )
    cb = base_callbacks
    extras = []

    composite_cb = CompositeCallback(update_manager, writers, cfg)
    extras.append(composite_cb)

    periodic_cb = PeriodicSnapshotsCallback(
        run_id=run_id,
        symbol=args.symbol,
        frame=args.frame,
        every_steps=int(getattr(args, "artifact_every_steps", 100_000)),
        vecnorm_applied=vecnorm_applied,
        writers=writers,
    )
    extras.append(periodic_cb)

    eval_cb: Optional[EvalCallback] = None
    if cfg.get("eval", {}).get("enable", True):
        eval_cb = EvalSaveCallback(
            eval_env,
            latest_path=paths["model_zip"],
            best_path=paths["model_best_zip"],
            vecnorm_path=paths["vecnorm_pkl"],
            vecnorm_best=paths["vecnorm_best"],
            best_meta=paths["best_meta"],
            vecnorm_ref=vecnorm_ref,
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

    # initial snapshot before training
    try:
        writers.train.last_step = 0
    except Exception:
        pass
    try:
        payload = {
            "symbol": args.symbol,
            "frame": args.frame,
            "ts": dt.datetime.utcnow().isoformat(),
            "global_step": 0,
            "status": "running",
            "vecnorm_applied": vecnorm_applied,
        }
        commit_snapshot(run_id, 0, payload)
    except Exception as e:
        logging.warning("[MEMORY] initial snapshot failed: %s", e)

    if getattr(args, "monitor", True):
        root_dir = str(paths_obj.root)
        spawn_monitor_manager(root_dir, args, run_id, headless=getattr(args, "headless", False))

    # 11) Learn
    status = "finished"
    best_path: Optional[str] = None
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
        saved_vec = False
        try:
            end_ts = int(getattr(model, "num_timesteps", 0))
            model.save(paths["model_zip"])  # final
            saved_vec = _save_vecnorm(vecnorm_ref, paths["vecnorm"], logging)
            logging.info("[SAVE] model -> %s | vecnorm -> %s", paths["model_zip"], paths["vecnorm"])
            best_path = _manage_models(paths, summary, run_id)
            run_meta.update({
                "vecnorm_applied": vecnorm_applied,
                "best_model_path": best_path,
                "best_model": best_path,
                "vecnorm_snapshot_saved": saved_vec,
            })
            paths_obj.write_run_meta({
                "vecnorm_applied": vecnorm_applied,
                "best_model_path": best_path,
                "best_model": best_path,
                "vecnorm_snapshot_saved": saved_vec,
            })
            summary["best_model_path"] = best_path
            summary["best_model"] = best_path
            try:
                meta = {"step": end_ts}
                if saved_vec:
                    meta["vecnorm_snapshot"] = True
                with open(paths["last_meta"], "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
            except Exception:
                pass
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
            log_queue.put(None)
        except Exception:
            pass
        try:
            log_thread.join()
        except Exception:
            pass
        logging.shutdown()
        try:
            writers.close()
        except Exception:
            pass

    # final snapshot after training
    try:
        step_now = int(getattr(model, "num_timesteps", 0))
        payload = {
            "symbol": args.symbol,
            "frame": args.frame,
            "ts": dt.datetime.utcnow().isoformat(),
            "global_step": step_now,
            "status": status,
            "vecnorm_applied": vecnorm_applied,
            "best_model_path": best_path,
            "best_model": best_path,
            "vecnorm_snapshot_saved": saved_vec,
        }
        commit_snapshot(run_id, step_now, payload)
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

    # synthetic evaluation of the run
    try:
        run_meta["synthetic_eval"] = evaluate_run(
            args.symbol,
            args.frame,
            run_id=run_id,
            episodes=int(run_meta.get("eval_episodes", 10)),
        )
    except Exception as e:
        logging.warning("[EVAL_RUN] failed: %s", e)

    summary_meta = _postrun_summary(paths, run_meta)

    if mon_proc is not None:
        try:
            mon_proc.terminate()
        except Exception:
            pass

    return True


# =============================
# Main (playlist or single job)
# =============================

def main():
    ensure_headless_once("train_rl")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    from bot_trade.config.rl_args import parse_args, finalize_args, build_policy_kwargs
    global torch, np, pd, psutil, subprocess, shutil

    args = parse_args()
    args.kb_file = str(Path(getattr(args, "kb_file", DEFAULT_KB_FILE) or DEFAULT_KB_FILE))

    device_str = normalize_device(getattr(args, "device", None), os.environ)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        if device_str and device_str.startswith("cuda"):
            print("[INFO] torch not available, falling back to CPU", flush=True)
            device_str = "cpu"
        else:
            print(
                "[ERROR] PyTorch is required for training.\n"
                "For CPU:   conda install pytorch cpuonly -c pytorch\n"
                "For CUDA:  conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia",
            )
            raise SystemExit(1)

    if device_str is None:
        device_str = (
            "cuda:0" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        )
    elif device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[INFO] CUDA requested but not available; using CPU", flush=True)
        device_str = "cpu"

    args.device_str = device_str
    logging.info("Using device=%s", device_str)
    args.policy_kwargs = build_policy_kwargs(args.net_arch, args.activation, args.ortho_init)
    global build_env_fns, make_vec_env, detect_action_space, build_ppo, build_sac, build_td3, build_tqc, build_callbacks
    global build_paths, ensure_state_files
    global Writers, create_loggers, setup_worker_logging, UpdateManager, CompositeCallback, get_config
    global EvalCallback, EvalSaveCallback, load_run_state, save_run_state, MemoryManager
    global load_memory, commit_snapshot, resume_from_snapshot, new_run_id
    global load_portfolio_state, save_portfolio_state, reset_with_balance
    global discover_files, read_one, LoadOptions, load_dataset, add_strategy_features, _HAS_READ_ONE
    global CallbackList, BenchmarkCallback, StrictDataSanityCallback, PeriodicSnapshotsCallback

    import math, psutil, numpy as np, pandas as pd, subprocess, shutil

    from bot_trade.config.rl_builders import (
        build_env_fns,
        make_vec_env,
        detect_action_space,
        build_ppo,
        build_sac,
        build_td3,
        build_tqc,
        build_callbacks,
    )
    from bot_trade.config.rl_paths import build_paths, ensure_state_files
    from bot_trade.config.rl_writers import Writers  # Writers bundle (train/eval/...)
    from bot_trade.config.log_setup import create_loggers, setup_worker_logging
    from bot_trade.config.update_manager import UpdateManager
    from bot_trade.config.rl_callbacks import CompositeCallback, PeriodicSnapshotsCallback
    from bot_trade.config.env_config import get_config
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import VecEnvWrapper, sync_envs_normalization
    from bot_trade.tools.run_state import load_state as load_run_state, save_state as save_run_state
    from bot_trade.tools.memory_manager import (
        MemoryManager,
        load_memory,
        commit_snapshot,
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
        from bot_trade.config.rl_callbacks import BenchmarkCallback, StrictDataSanityCallback, PeriodicSnapshotsCallback
    except Exception:  # pragma: no cover
        BenchmarkCallback = None
        StrictDataSanityCallback = None
        PeriodicSnapshotsCallback = None
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
            vecnorm_ref=None,
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
            self._vecnorm_ref = vecnorm_ref
            self._last_mtime = 0.0
            self._patience = int(patience)
            self._lr_factor = float(lr_factor)
            self._lr_limit = int(lr_limit)
            self._no_improve = 0
            self._lr_updates = 0
            self._best_metric = -float("inf")

        def _on_step(self) -> bool:  # type: ignore[override]
            run_eval = self.eval_freq and self.n_calls % self.eval_freq == 0
            if run_eval:
                orig_get = self.model.get_vec_normalize_env
                try:
                    if isinstance(self.training_env, VecEnvWrapper) and isinstance(self.eval_env, VecEnvWrapper):
                        try:
                            sync_envs_normalization(self.training_env, self.eval_env)
                        except Exception:
                            logging.warning("[Eval] normalization sync failed", exc_info=True)
                    self.model.get_vec_normalize_env = lambda *_, **__: None
                    result = super()._on_step()
                finally:
                    self.model.get_vec_normalize_env = orig_get
            else:
                result = super()._on_step()
            if run_eval:
                try:
                    self.model.save(self._latest_path)
                    _save_vecnorm(self._vecnorm_ref, self._vecnorm_path, logging)
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
                                "vecnorm_snapshot": True,
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
    try:
        files = discover_files(args.frame, args.symbol, data_root=args.data_root)
    except FileNotFoundError:
        if getattr(args, "allow_synth", False):
            from bot_trade.tools.gen_synth_data import generate

            base = Path(args.data_root or "data_ready")
            dest = generate(args.symbol, args.frame, base)
            print(f"[SYNTH] generated {dest}")
            files = discover_files(args.frame, args.symbol, data_root=args.data_root)
        else:
            base = pathlib.Path(args.data_root) / args.frame
            pats = [
                f"{base}/{args.symbol}-{args.frame}-*.feather",
                f"{base}/{args.symbol}-{args.frame}-*.parquet",
                f"{base}/{args.symbol}-{args.frame}-*.csv",
            ]
            print("[NO DATA] Searched paths:", file=sys.stderr)
            for p in pats:
                print(f"  - {p}", file=sys.stderr)
            raise SystemExit(3)
    if not files:
        base = pathlib.Path(args.data_root) / args.frame
        pats = [
            f"{base}/{args.symbol}-{args.frame}-*.feather",
            f"{base}/{args.symbol}-{args.frame}-*.parquet",
            f"{base}/{args.symbol}-{args.frame}-*.csv",
        ]
        print("[NO DATA] Searched paths:", file=sys.stderr)
        for p in pats:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(3)
    data_file = files[0]
    logging.info("[DATA] using file %s", data_file)
    train_one_file(args, data_file)
    mm.snapshot({"data_file": data_file, "origin": args.data_origin})
    mm.log_event("end", {"status": "ok"})


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
