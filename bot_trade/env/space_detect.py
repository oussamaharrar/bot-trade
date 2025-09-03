from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from gymnasium import spaces as gym_spaces


@dataclass(frozen=True)
class ActionSpaceInfo:
    is_discrete: bool
    shape: Tuple[int, ...]
    low: Optional[float]
    high: Optional[float]


def detect_action_space(env: Any) -> ActionSpaceInfo:
    """Detect basic properties of an environment's action space.

    ``env`` may be an environment or a Gymnasium space instance.
    """

    space = getattr(env, "single_action_space", None)
    if space is None:
        space = getattr(env, "action_space", None)
    if space is None:
        space = env
    if isinstance(space, gym_spaces.Discrete):
        n = int(getattr(space, "n", 0))
        return ActionSpaceInfo(True, (1,), 0.0, float(n - 1))
    if isinstance(space, gym_spaces.Box):
        shape = tuple(space.shape)
        low_arr = np.asarray(space.low, dtype=float)
        high_arr = np.asarray(space.high, dtype=float)
        low_val = float(low_arr.flat[0]) if np.all(low_arr == low_arr.flat[0]) else None
        high_val = float(high_arr.flat[0]) if np.all(high_arr == high_arr.flat[0]) else None
        return ActionSpaceInfo(False, shape, low_val, high_val)
    shape = tuple(getattr(space, "shape", ()) or ())
    return ActionSpaceInfo(False, shape, None, None)


__all__ = ["ActionSpaceInfo", "detect_action_space"]
