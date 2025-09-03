"""Deprecated action space detection shim."""

from __future__ import annotations

from typing import Any

from .space_detect import ActionSpaceInfo
from .space_detect import detect_action_space as _detect_action_space

_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        print("[DEPRECATION] bot_trade.env.action_space is deprecated; use bot_trade.env.space_detect")
        _WARNED = True


_warn_once()


class _Wrap:
    def __init__(self, space: Any) -> None:
        self.action_space = space


def detect_action_space(env: Any) -> ActionSpaceInfo:
    """Legacy wrapper for :func:`bot_trade.env.space_detect.detect_action_space`."""

    if hasattr(env, "action_space"):
        info_env = _detect_action_space(env)
        info_space = _detect_action_space(env.action_space)
        assert info_env == info_space
        return info_env
    return _detect_action_space(_Wrap(env))


__all__ = ["ActionSpaceInfo", "detect_action_space"]
