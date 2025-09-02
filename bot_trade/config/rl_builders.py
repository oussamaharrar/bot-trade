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
import os, logging
from pathlib import Path

from .rl_paths import best_agent, last_agent, get_root

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3 import PPO, SAC

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
) -> Callable[[], TradingEnv]:
    """
    Top-level factory that returns a picklable _init() for SB3 VecEnvs.
    NOTE:
      - When used under SubprocVecEnv, pass writers=None to avoid Windows pipe errors.
      - When using DummyVecEnv (n_envs=1), you may pass writers to enable main-process logging.
    """
    def _init():
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


def detect_action_space(vec_env):
    """Safely detect action space and whether it's discrete."""
    try:
        action_space = getattr(vec_env, "single_action_space", None)
        if action_space is None:
            action_space = getattr(vec_env, "action_space", None)
        from gymnasium import spaces as gym_spaces
        is_discrete = isinstance(action_space, gym_spaces.Discrete)
        return action_space, is_discrete
    except Exception:
        return None, False


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


def build_ppo(env, args, policy_kwargs):
    action_space, is_discrete = detect_action_space(env)
    use_sde = bool(args.sde and (not is_discrete))
    if args.sde and is_discrete:
        logging.warning("[PPO] gSDE disabled automatically for Discrete action space.")

    bs = _adjust_batch_size_for_envs(int(getattr(args, "batch_size", 64)), env)
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
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        sde_sample_freq=4 if use_sde else -1,
        seed=args.seed,
        device=args.device_str,
        verbose=getattr(args, "sb3_verbose", 1),
        tensorboard_log=getattr(args, "tensorboard_log", None),
    )
    return model


def build_sac(env, args, policy_kwargs):
    """Return a configured SAC model (from stable_baselines3)."""
    action_space, _ = detect_action_space(env)
    from gymnasium import spaces as gym_spaces

    if not isinstance(action_space, gym_spaces.Box):
        msg = f"[ALGO] SAC requires a continuous (Box) action space, got {type(action_space).__name__}"
        logging.getLogger(__name__).error(msg)
        print(msg, flush=True)
        raise SystemExit(1)

    from .env_config import get_config

    cfg = get_config()
    rl_cfg = cfg.get("rl", {}) if isinstance(cfg, dict) else {}
    sac_cfg = rl_cfg.get("sac", {}) if isinstance(rl_cfg, dict) else {}

    overrides: dict[str, object] = {}
    defaults = getattr(args, "_defaults", {})
    specified = getattr(args, "_specified", set())

    def _get(name, default):
        val = getattr(args, name, None)
        if name in specified and val is not None:
            overrides[name] = val
            return val
        if val is not None and val != defaults.get(name):
            return val
        return sac_cfg.get(name, rl_cfg.get(name, default))

    buffer_size = int(_get("buffer_size", 2_000_000))
    learning_starts = int(_get("learning_starts", 20_000))
    train_freq = int(_get("train_freq", 1))
    gradient_steps = int(_get("gradient_steps", 1))
    tau = float(_get("tau", 0.005))
    ent_coef = _get("ent_coef", "auto")
    try:
        ent_coef = float(ent_coef)
    except (TypeError, ValueError):
        pass
    gamma = _get("sac_gamma", None)
    if gamma is None:
        gamma = _get("gamma", 0.99)
    learning_rate = _get("learning_rate", 3e-4)

    bs_default = int(_get("batch_size", 512))
    bs = _adjust_batch_size_for_envs(bs_default, env)

    if overrides:
        msg = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"[ALGO] SAC overrides: {msg}")

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        batch_size=bs,
        tau=tau,
        ent_coef=ent_coef,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device_str,
        verbose=getattr(args, "sb3_verbose", 1),
        tensorboard_log=getattr(args, "tensorboard_log", None),
    )

    if getattr(args, "sac_warmstart_from_ppo", False) or getattr(args, "warmstart_from_ppo", None):
        ppo_path = _resolve_ppo_warmstart(args)
        if ppo_path:
            try:
                from stable_baselines3 import PPO as _PPO

                _ppo = _PPO.load(str(ppo_path), device=args.device_str)
                model.policy.features_extractor.load_state_dict(
                    _ppo.policy.features_extractor.state_dict(), strict=False
                )
                logging.getLogger(__name__).info(
                    "[ALGO] warm-started SAC feature extractor from %s", ppo_path
                )
            except Exception as e:
                logging.getLogger(__name__).warning("[ALGO] warm-start skipped: %s", e)
        else:
            msg = "[ALGO] warm-start skipped: compatible PPO checkpoint not found"
            print(msg)
            logging.getLogger(__name__).info(msg)

    return model


def build_td3_stub(env, args, policy_kwargs):  # noqa: ARG001
    print("[ALGO] TD3 not implemented yet", flush=True)
    raise SystemExit(2)


def build_tqc_stub(env, args, policy_kwargs):  # noqa: ARG001
    print("[ALGO] TQC not implemented yet", flush=True)
    raise SystemExit(2)


REGISTRY = {
    "PPO": build_ppo,
    "SAC": build_sac,
    "TD3": build_td3_stub,
    "TQC": build_tqc_stub,
}


def build_algorithm(name: str, env, args, policy_kwargs):
    return REGISTRY[name.upper()](env, args, policy_kwargs)


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
