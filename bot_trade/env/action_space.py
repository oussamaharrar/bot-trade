"""Deprecated action space detection shim."""

from __future__ import annotations

from typing import Any

from .space_detect import ActionSpaceInfo, detect_action_space as _detect_action_space

_WARNED = False


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        print(
            "[DEPRECATION] bot_trade.env.action_space is deprecated; use bot_trade.env.space_detect"
        )
        _WARNED = True


_warn_once()


def detect_action_space(space: Any) -> ActionSpaceInfo:
    """Legacy wrapper for :func:`bot_trade.env.space_detect.detect_action_space`."""

    return _detect_action_space(space)


__all__ = ["ActionSpaceInfo", "detect_action_space"]
