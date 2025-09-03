"""Action space detection utilities."""

from __future__ import annotations

from typing import Any, List

from gymnasium import spaces as gym_spaces


def detect_action_space(env: Any) -> dict:
    """Return action space details for ``env``.

    The result dictionary contains:

    ``kind``
        ``"discrete"`` for :class:`gymnasium.spaces.Discrete`, ``"box"`` for
        :class:`gymnasium.spaces.Box`, or the lowercase class name otherwise.
    ``shape``
        List of ``int`` describing the action shape.
    ``low`` / ``high``
        Bounds for ``Box`` spaces, empty lists otherwise.
    """

    space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    if isinstance(space, gym_spaces.Discrete):
        kind = "discrete"
    elif isinstance(space, gym_spaces.Box):
        kind = "box"
    else:
        kind = type(space).__name__.lower()
    shape: List[int] = list(getattr(space, "shape", []) or [])
    low = space.low.tolist() if isinstance(space, gym_spaces.Box) else []
    high = space.high.tolist() if isinstance(space, gym_spaces.Box) else []
    return {"kind": kind, "shape": shape, "low": low, "high": high}


__all__ = ["detect_action_space"]
