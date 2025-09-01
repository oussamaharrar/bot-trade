"""Deprecated shim for :mod:`eval_run`."""
from __future__ import annotations

import sys

from bot_trade.tools.eval_run import evaluate_run as evaluate_for_run, main as _main

_WARNED = False


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    global _WARNED
    if not _WARNED:
        print("[DEPRECATED] use bot_trade.tools.eval_run", file=sys.stderr)
        _WARNED = True
    argv = list(argv or sys.argv[1:])
    cleaned: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == "--base":
            skip = True
            continue
        cleaned.append(arg)
    return _main(cleaned)

__all__ = ["evaluate_for_run", "main"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
