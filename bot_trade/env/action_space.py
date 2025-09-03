"""Action space detection utilities."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from gymnasium import spaces as gym_spaces


def detect_action_space(env: Any) -> dict:
    """Return ``{is_discrete, shape, low, high}`` for ``env``'s action space."""

    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    is_discrete = isinstance(space, gym_spaces.Discrete)
    if is_discrete:
        shape: Tuple[int, ...] = (int(getattr(space, "n", 0)),)
        low = np.array([])
        high = np.array([])
    elif isinstance(space, gym_spaces.Box):
        shape = tuple(space.shape)
        low = np.asarray(space.low)
        high = np.asarray(space.high)
    else:
        shape = tuple(getattr(space, "shape", ()) or ())
        low = np.array([])
        high = np.array([])
    return {"is_discrete": is_discrete, "shape": shape, "low": low, "high": high}


__all__ = ["detect_action_space"]
