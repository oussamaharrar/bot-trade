from __future__ import annotations

"""Action space detection utilities."""

from typing import Any, List

from gymnasium import spaces as gym_spaces


def detect_action_space(env: Any) -> dict:
    """Return action space details for ``env``."""

    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    is_discrete = isinstance(space, gym_spaces.Discrete)
    shape: List[int] = list(space.shape) if getattr(space, "shape", None) else []
    low = space.low.tolist() if isinstance(space, gym_spaces.Box) else []
    high = space.high.tolist() if isinstance(space, gym_spaces.Box) else []
    return {"is_discrete": is_discrete, "shape": shape, "low": low, "high": high}
