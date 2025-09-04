from __future__ import annotations

"""Minimal RL agent builders wrapping Stable-Baselines3 algorithms."""

from typing import Any, Dict, Tuple

from gymnasium import spaces
import numpy as np


import gymnasium as gym

class _DummyEnv(gym.Env):
    metadata = {"render_modes": []}
    """Simple one-step environment with Box spaces."""

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # noqa: D401
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):  # noqa: D401
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


def _require_box(space) -> None:
    if not isinstance(space, spaces.Box):
        raise TypeError("action space must be gymnasium.spaces.Box")


def collect_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Stable-Baselines3 policy kwargs overrides from *cfg*."""

    return {k.replace("policy_", "", 1): v for k, v in cfg.items() if k.startswith("policy_")}


def build_ppo(cfg: Dict[str, Any]):
    from stable_baselines3 import PPO

    env = cfg.get("env") or _DummyEnv()
    _require_box(env.action_space)
    policy_kwargs = collect_overrides(cfg)
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    return model, env, []


def build_sac(cfg: Dict[str, Any]):
    from stable_baselines3 import SAC

    env = cfg.get("env") or _DummyEnv()
    _require_box(env.action_space)
    policy_kwargs = collect_overrides(cfg)
    model = SAC("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    return model, env, []


def build_td3(cfg: Dict[str, Any]):
    from stable_baselines3 import TD3

    env = cfg.get("env") or _DummyEnv()
    _require_box(env.action_space)
    policy_kwargs = collect_overrides(cfg)
    model = TD3("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    return model, env, []


def build_tqc(cfg: Dict[str, Any]):
    from stable_baselines3 import TQC  # type: ignore

    env = cfg.get("env") or _DummyEnv()
    _require_box(env.action_space)
    policy_kwargs = collect_overrides(cfg)
    model = TQC("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    return model, env, []


_BUILDERS = {
    "PPO": build_ppo,
    "SAC": build_sac,
    "TD3": build_td3,
    "TQC": build_tqc,
}


def build_agent(algorithm: str, cfg: Dict[str, Any]) -> Tuple[Any, Any, list]:
    """Return ``(model, env, callbacks)`` for *algorithm* using *cfg*."""

    algo = algorithm.upper()
    if algo not in _BUILDERS:
        raise KeyError(algo)
    return _BUILDERS[algo](cfg)


__all__ = [
    "_require_box",
    "collect_overrides",
    "build_ppo",
    "build_sac",
    "build_td3",
    "build_tqc",
    "build_agent",
]
