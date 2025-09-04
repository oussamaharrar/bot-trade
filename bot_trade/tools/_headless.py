from __future__ import annotations

"""Compatibility wrapper for deprecated headless helper."""

from bot_trade.strutils.headless import ensure_headless_once as _ensure, HEADLESS_MARK  # noqa: F401

def ensure_headless_once(*_args, **_kwargs):
    _ensure()
