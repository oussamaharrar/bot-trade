# rl_builders.py
"""
Constructors for vectorized RL environments (SB3) with Windows-safe subprocessing.

Key points:
- Use DummyVecEnv when n_envs == 1 (no IPC, safer on Windows).
- Never pass IO handles (e.g., writers) into subprocess envs.
- Provide top-level factory functions so cloudpickle/spawn works on Windows.
- Optionally enable VecNormalize.
"""

from __future__ import annotations
from typing import Callable, List, Optional
import os, logging, json, hashlib
from pathlib import Path

from .rl_paths import best_agent, last_agent, get_root
from bot_trade.env.action_space import detect_action_space

_ARGS_WARNED = False
_GLOBAL_IGNORE = {
    "symbol",
    "frame",
    "device",
    "n_envs",
    "total_steps",
    "headless",
    "allow_synth",
    "data_dir",
    "no_monitor",
    "algorithm",
    "continuous_env",
    "policy",
    "policy_kwargs",
    "preset",
    "kb_file",
    "run_id",
    "seed",
    "device_str",
    "log_level",
}


def collect_overrides(args, valid: set[str]) -> dict:
    defaults = getattr(args, "_defaults", {})
    specified = getattr(args, "_specified", set())
    overrides = {}
    for k in valid:
        if k in specified and getattr(args, k) != defaults.get(k):
            overrides[k] = getattr(args, k)
    unused = specified - valid - _GLOBAL_IGNORE
    global _ARGS_WARNED
    if unused and not _ARGS_WARNED:
        print(f"[ARGS_WARN] unused={','.join(sorted(unused))}")
        _ARGS_WARNED = True
    return overrides


def _hash_overrides(overrides: dict) -> str:
    if not overrides:
        return ""
    js = json.dumps(overrides, sort_keys=True)
    return hashlib.sha1(js.encode("utf-8")).hexdigest()[:8]

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

# استيراد بيئتك
from .env_trading import TradingEnv


def make_env_ctor(
    df,
    frame: str,
    symbol: str,
    use_indicators: bool = True,
    config: Optional[dict] = None,
    # لا تمرّر writers إلى العمال عند التوازي
    writers=None,
    safe: bool = True,
    decisions_jsonl: Optional[str] = None,
    continuous: bool = False,
) -> Callable[[], TradingEnv]:
    """
    Top-level factory that returns a picklable _init() for SB3 VecEnvs.
    NOTE:
      - When used under SubprocVecEnv, pass writers=None to avoid Windows pipe errors.
      - When using DummyVecEnv (n_envs=1), you may pass writers to enable main-process logging.
    """
    def _init():
        if continuous:
            from bot_trade.env.trading_env_continuous import TradingEnvContinuous

            env = TradingEnvContinuous(
                data=df,
                frame=frame,
                symbol=symbol,
                use_indicators=use_indicators,
                config=config,
                writers=writers,
                decisions_jsonl=decisions_jsonl,
                safe=safe,
            )
        else:
            env = TradingEnv(
                data=df,
                frame=frame,
                symbol=symbol,
                use_indicators=use_indicators,
                config=config,
                writers=writers,
                decisions_jsonl=decisions_jsonl,
                safe=safe,
            )
        return env
    return _init


def build_env_fns(
    df,
    frame: str,
    symbol: str,
    n_envs: int,
    use_indicators: bool = True,
    config: Optional[dict] = None,
    writers=None,
    safe: bool = True,
    decisions_jsonl: Optional[str] = None,
    pass_writers_when_single: bool = True,
    continuous: bool = False,
) -> List[Callable[[], TradingEnv]]:
    """
    Build a list of environment constructors. For n_envs > 1 we DO NOT pass writers to subprocs.
    For n_envs == 1 we pass writers only if pass_writers_when_single=True (recommended).
    """
    env_fns: List[Callable[[], TradingEnv]] = []
    for i in range(n_envs):
        writers_for_this = writers if (n_envs == 1 and pass_writers_when_single) else None
        ctor = make_env_ctor(
            df=df,
            frame=frame,
            symbol=symbol,
            use_indicators=use_indicators,
            config=config,
            writers=writers_for_this,
            safe=safe,
            decisions_jsonl=decisions_jsonl,
            continuous=continuous,
        )
        env_fns.append(ctor)
    return env_fns


def make_vec_env(
    env_fns: List[Callable[[], TradingEnv]],
    n_envs: int,
    start_method: str = "spawn",
    normalize: bool = True,
    seed: Optional[int] = None,
) -> VecEnv:
    """
    Create a vectorized environment:
      - DummyVecEnv if n_envs == 1
      - SubprocVecEnv otherwise (with explicit start_method, default 'spawn')
    Optionally wraps with VecNormalize.
    """
    # DummyVecEnv محلي (آمن على ويندوز)
    if n_envs == 1:
        vec: VecEnv = DummyVecEnv(env_fns)
        if seed is not None:
            try:
                vec.seed(seed)
            except Exception:
                pass
    else:
        # SubprocVecEnv عبر spawn يخفف مشاكل pickling/IPC على ويندوز
        vec = SubprocVecEnv(env_fns, start_method=start_method)
        if seed is not None:
            try:
                # بذور مختلفة لكل عامل
                base = int(seed)
                vec.seed([base + i for i in range(n_envs)])
            except Exception:
                pass

    if normalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return vec


def load_vecnormalize_safe(vec_env: VecNormalize, *paths: str) -> bool:
    """تحميل إحصاءات VecNormalize بأمان إن توفرت بأي من المسارات."""
    loaded = False
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p) and hasattr(vec_env, "load_running_average"):
                vec_env.load_running_average(p)
                logging.info("[VECNORM] loaded running average from %s", p)
                loaded = True
                break
        except Exception as e:
            logging.warning("[VECNORM] failed to load %s: %s", p, e)
    if not loaded:
        logging.info("[VECNORM] no previous running average found.")
    return loaded


def _adjust_batch_size_for_envs(batch_size: int, vec_env) -> int:
    """اجعل batch_size من مضاعفات n_envs لتفادي تحذيرات/أخطاء SB3."""
    try:
        n_envs = int(getattr(vec_env, "num_envs", 1))
    except Exception:
        n_envs = 1
    if n_envs <= 1:
        return int(batch_size)
    if batch_size % n_envs == 0:
        return int(batch_size)
    new_bs = (batch_size // n_envs) * n_envs
    new_bs = max(new_bs, n_envs)  # لا نصل للصفر
    if new_bs != batch_size:
        logging.warning("[PPO] batch_size=%d ليس من مضاعفات n_envs=%d → ضبطه إلى %d", batch_size, n_envs, new_bs)
    return int(new_bs)




def _resolve_ppo_warmstart(args) -> Path | None:
    """Return first readable PPO checkpoint for warm-start."""
    candidates = []
    explicit = getattr(args, "warmstart_from_ppo", None)
    if explicit:
        p = Path(explicit)
        if p.exists():
            candidates.append(p)
    candidates.append(best_agent(args.symbol, args.frame, "PPO"))
    candidates.append(last_agent(args.symbol, args.frame, "PPO"))
    legacy = get_root() / "agents" / args.symbol.upper() / str(args.frame)
    candidates.extend([
        legacy / "deep_rl_best.zip",
        legacy / "deep_rl_last.zip",
        legacy / "deep_rl.zip",
    ])
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

_policy_json_warned = False
_policy_unused_warned = False


def _policy_kwargs_from_args(args) -> tuple[dict, set]:
    global _policy_json_warned
    pk = getattr(args, "policy_kwargs", {}) or {}
    if isinstance(pk, str):
        try:
            pk = json.loads(pk)
        except Exception:
            if not _policy_json_warned:
                print("[ARGS] invalid policy_kwargs JSON")
                _policy_json_warned = True
            pk = {}
    if not isinstance(pk, dict):
        pk = {}
    user_keys = set(pk.keys())
    act = pk.get("activation_fn")
    if isinstance(act, str):
        try:
            from torch import nn

            mapping = {
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "silu": nn.SiLU,
                "elu": nn.ELU,
                "gelu": nn.GELU,
            }
            pk["activation_fn"] = mapping.get(act.lower(), nn.ReLU)
        except Exception:
            pk.pop("activation_fn", None)
    return pk, user_keys


def _warn_unused_policy_kwargs(user_keys: set, used: dict) -> None:
    global _policy_unused_warned
    unused = sorted(user_keys - set(used.keys()))
    if unused and not _policy_unused_warned:
        print(f"[ARGS] unused policy_kwargs keys={unused}")
        _policy_unused_warned = True


def _condense_policy_kwargs(pk: dict) -> dict:
    out: dict = {}
    for k, v in pk.items():
        if hasattr(v, "__name__"):
            out[k] = v.__name__
        elif isinstance(v, (list, tuple)):
            if len(v) > 4:
                out[k] = list(v[:3]) + ["..."]
            else:
                out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = {sk: (len(sv) if isinstance(sv, (list, tuple)) else sv) for sk, sv in v.items()}
        else:
            out[k] = v
    return out


def _require_box(env, name: str) -> None:
    info = detect_action_space(env)
    if info["is_discrete"] or not info["low"]:
        print(
            f"[ALGO_GUARD] algorithm={name} requires continuous Box action space; got {type(getattr(env, 'action_space', None)).__name__}. Aborting.",
            flush=True,
        )
        raise SystemExit(1)


def build_ppo(env, args, seed):
    from stable_baselines3 import PPO

    pk, pk_keys = _policy_kwargs_from_args(args)
    bs = _adjust_batch_size_for_envs(int(getattr(args, "batch_size", 64)), env)
    info = detect_action_space(env)
    use_sde = bool(getattr(args, "sde", False) and not info["is_discrete"])
    if getattr(args, "sde", False) and info["is_discrete"]:
        logging.warning("[PPO] gSDE disabled automatically for Discrete action space.")
    ent_coef = float(args.ent_coef) if isinstance(args.ent_coef, str) else args.ent_coef
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=bs,
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=pk,
        use_sde=use_sde,
        sde_sample_freq=4 if use_sde else -1,
        seed=seed,
        device=args.device_str,
        verbose=getattr(args, "sb3_verbose", 1),
        tensorboard_log=getattr(args, "tensorboard_log", None),
    )
    _warn_unused_policy_kwargs(pk_keys, model.policy_kwargs)
    valid = {
        "learning_rate",
        "n_steps",
        "batch_size",
        "epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "sde",
    }
    overrides = collect_overrides(args, valid)
    meta = {
        "lr": args.learning_rate,
        "batch_size": bs,
        "buffer_size": None,
        "gamma": args.gamma,
        "policy_kwargs": _condense_policy_kwargs(pk),
        "overrides": overrides,
        "overrides_hash": _hash_overrides(overrides),
    }
    return model, meta


def build_sac(env, args, seed):
    from stable_baselines3 import SAC
    _require_box(env, "SAC")
    pk, pk_keys = _policy_kwargs_from_args(args)
    pk = pk.copy()
    pk.pop("ortho_init", None)
    if isinstance(pk.get("net_arch"), dict):
        pk["net_arch"] = pk["net_arch"].get("pi")
    buffer_size = int(getattr(args, "buffer_size", None) or 100_000)
    learning_starts = int(getattr(args, "learning_starts", None) or 1_000)
    batch_size = _adjust_batch_size_for_envs(int(getattr(args, "batch_size", None) or 256), env)
    gamma = float(getattr(args, "gamma", None) or 0.99)
    tau = float(getattr(args, "tau", None) or 0.005)
    ent_coef = getattr(args, "ent_coef", "auto")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=getattr(args, "learning_rate", 3e-4),
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=ent_coef,
        target_entropy="auto",
        train_freq=(1, "step"),
        gradient_steps=int(getattr(args, "gradient_steps", None) or 1),
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=pk,
        seed=seed,
        device=args.device_str,
        verbose=getattr(args, "sb3_verbose", 1),
        tensorboard_log=getattr(args, "tensorboard_log", None),
    )
    ppo_path = _resolve_ppo_warmstart(args)
    if ppo_path:
        try:
            from stable_baselines3 import PPO as _PPO

            _ppo = _PPO.load(str(ppo_path), device=args.device_str)
            src = _ppo.policy.features_extractor.state_dict()
            dst = model.policy.features_extractor.state_dict()
            if all(k in dst and v.shape == dst[k].shape for k, v in src.items()):
                dst.update({k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape})
                model.policy.features_extractor.load_state_dict(dst, strict=False)
                print(
                    f"[WARM_START] source=PPO checkpoint={ppo_path} layers={len(dst)} status=applied"
                )
            else:
                print("[WARM_START] source=PPO status=skipped reason=mismatch")
        except Exception:
            print("[WARM_START] source=PPO status=skipped reason=mismatch")
    else:
        print("[WARM_START] source=PPO status=skipped reason=not_found")
    _warn_unused_policy_kwargs(pk_keys, model.policy_kwargs)
    meta = {
        "lr": getattr(args, "learning_rate", 3e-4),
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "policy_kwargs": _condense_policy_kwargs(pk),
    }
    return model, meta


def build_td3(env, args, seed):
    from stable_baselines3 import TD3
    _require_box(env, "TD3")
    pk, pk_keys = _policy_kwargs_from_args(args)
    pk = pk.copy()
    pk.pop("ortho_init", None)
    if isinstance(pk.get("net_arch"), dict):
        pk["net_arch"] = pk["net_arch"].get("pi")
    buffer_size = int(getattr(args, "buffer_size", None) or 100_000)
    learning_starts = int(getattr(args, "learning_starts", None) or 1_000)
    batch_size = _adjust_batch_size_for_envs(int(getattr(args, "batch_size", None) or 256), env)
    gamma = float(getattr(args, "gamma", None) or 0.99)
    tau = float(getattr(args, "tau", None) or 0.005)
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=getattr(args, "learning_rate", 3e-4),
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        train_freq=(1, "step"),
        gradient_steps=int(getattr(args, "gradient_steps", None) or 1),
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs=pk,
        seed=seed,
        device=args.device_str,
        verbose=getattr(args, "sb3_verbose", 1),
        tensorboard_log=getattr(args, "tensorboard_log", None),
    )
    _warn_unused_policy_kwargs(pk_keys, model.policy_kwargs)
    valid = {
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_freq",
        "gradient_steps",
        "tau",
        "ent_coef",
        "gamma",
        "target_entropy",
    }
    overrides = collect_overrides(args, valid)
    meta = {
        "lr": getattr(args, "learning_rate", 3e-4),
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "policy_kwargs": _condense_policy_kwargs(pk),
        "overrides": overrides,
        "overrides_hash": _hash_overrides(overrides),
    }
    return model, meta


def build_tqc(env, args, seed):
    _require_box(env, "TQC")
    try:
        from sb3_contrib import TQC
    except Exception:
        print("[ALGO_GUARD] algorithm=TQC unavailable (missing sb3-contrib). Aborting.")
        raise SystemExit(1)
    pk, pk_keys = _policy_kwargs_from_args(args)
    pk = pk.copy()
    pk.pop("ortho_init", None)
    if isinstance(pk.get("net_arch"), dict):
        pk["net_arch"] = pk["net_arch"].get("pi")
    buffer_size = int(getattr(args, "buffer_size", None) or 100_000)
    learning_starts = int(getattr(args, "learning_starts", None) or 1_000)
    batch_size = _adjust_batch_size_for_envs(int(getattr(args, "batch_size", None) or 256), env)
    gamma = float(getattr(args, "gamma", None) or 0.99)
    tau = float(getattr(args, "tau", None) or 0.005)
    model = TQC(
        "MlpPolicy",
        env,
        learning_rate=getattr(args, "learning_rate", 3e-4),
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        ent_coef=getattr(args, "ent_coef", "auto"),
        target_entropy="auto",
        train_freq=(1, "step"),
        gradient_steps=int(getattr(args, "gradient_steps", None) or 1),
        use_sde=True,
        sde_sample_freq=4,
        n_critics=5,
        n_quantiles=25,
        top_quantiles_to_drop_per_net=2,
        policy_kwargs=pk,
        seed=seed,
        device=args.device_str,
        verbose=getattr(args, "sb3_verbose", 1),
        tensorboard_log=getattr(args, "tensorboard_log", None),
    )
    _warn_unused_policy_kwargs(pk_keys, model.policy_kwargs)
    valid = {
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "gamma",
        "tau",
        "gradient_steps",
        "ent_coef",
    }
    overrides = collect_overrides(args, valid)
    meta = {
        "lr": getattr(args, "learning_rate", 3e-4),
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "policy_kwargs": _condense_policy_kwargs(pk),
        "overrides": overrides,
        "overrides_hash": _hash_overrides(overrides),
    }
    return model, meta


REGISTRY = {
    "PPO": build_ppo,
    "SAC": build_sac,
    "TD3": build_td3,
    "TQC": build_tqc,
}


def build_algorithm(name: str, env, args, seed: int):
    fn = REGISTRY.get(name.upper())
    if not fn:
        raise ValueError(f"Unknown algorithm: {name}")
    return fn(env, args, seed)


def build_callbacks(
    paths,
    writers,
    args,
    update_manager=None,
    run_id=None,
    risk_manager=None,
    dataset_info=None,
    vecnorm_ref=None,
):
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from .rl_callbacks import StepsAndRewardCallback, BestCheckpointCallback
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, int(getattr(args, "checkpoint_every", 50_000))),
        save_path=paths["agents"],
        name_prefix="checkpoint",
    )
    step_cb = StepsAndRewardCallback(args.frame, args.symbol, writers, log_every=int(getattr(args, "log_every", 2_000)))

    best_cb = BestCheckpointCallback(paths, check_every=int(getattr(args, "best_check_every", 50_000)), vecnorm_ref=vecnorm_ref)

    callbacks = [ckpt_cb, step_cb, best_cb]

    try:
        from .rl_callbacks import BenchmarkCallback, StrictDataSanityCallback
        if getattr(args, "enable_benchmark", True):
            callbacks.append(BenchmarkCallback(frame=args.frame, symbol=args.symbol, writers=writers, every_sec=int(getattr(args, "bench_every_sec", 15))))
        if bool(getattr(args, "safe", False)):
            callbacks.append(StrictDataSanityCallback(writers=writers, raise_on_issue=True))
    except Exception as e:
        logging.debug("[CB] optional callbacks not available: %s", e)

    return CallbackList(callbacks)
