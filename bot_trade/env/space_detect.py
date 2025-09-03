from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from gymnasium import spaces as gym_spaces


@dataclass(frozen=True)
class ActionSpaceInfo:
    is_discrete: bool
    shape: Tuple[int, ...]
    low: np.ndarray
    high: np.ndarray


def detect_action_space(space: Any) -> ActionSpaceInfo:
    """Detect basic properties of an environment's action space.

    ``space`` may be an environment or a Gymnasium space instance.
    """

    s = getattr(space, "single_action_space", None) or getattr(space, "action_space", space)
    if isinstance(s, gym_spaces.Discrete):
        n = int(getattr(s, "n", 0))
        low = np.array([0], dtype=np.int64)
        high = np.array([n - 1], dtype=np.int64)
        return ActionSpaceInfo(True, (1,), low, high)
    if isinstance(s, gym_spaces.Box):
        return ActionSpaceInfo(False, tuple(s.shape), np.asarray(s.low), np.asarray(s.high))
    shape = tuple(getattr(s, "shape", ()) or ())
    return ActionSpaceInfo(False, shape, np.array([]), np.array([]))


__all__ = ["ActionSpaceInfo", "detect_action_space"]
