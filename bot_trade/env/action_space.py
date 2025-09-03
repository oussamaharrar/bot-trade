from __future__ import annotations

"""Action space detection utilities."""

from dataclasses import dataclass
from typing import Any

from gymnasium import spaces as gym_spaces


@dataclass
class ActionSpaceInfo:
    is_box: bool
    is_discrete: bool
    dims: int
    low: list[float]
    high: list[float]


def detect_action_space(env: Any) -> ActionSpaceInfo:
    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    is_box = isinstance(space, gym_spaces.Box)
    is_discrete = isinstance(space, gym_spaces.Discrete)
    try:
        dims = int(space.shape[0]) if getattr(space, "shape", None) else 0
    except Exception:
        dims = 0
    low = space.low.tolist() if is_box else []
    high = space.high.tolist() if is_box else []
    return ActionSpaceInfo(is_box=is_box, is_discrete=is_discrete, dims=dims, low=low, high=high)
