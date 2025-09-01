"""Deprecated shim for :mod:`export_charts`."""
from __future__ import annotations

import sys
from typing import Any, Dict, Tuple

from bot_trade.tools.export_charts import (
    export_run_charts,
    export_for_run,
    main as _main,
)

_WARNED = False


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    global _WARNED
    if not _WARNED:
        print("[DEPRECATED] use bot_trade.tools.export_charts", file=sys.stderr)
        _WARNED = True
    return _main(argv)

__all__ = ["export_run_charts", "export_for_run", "main"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
