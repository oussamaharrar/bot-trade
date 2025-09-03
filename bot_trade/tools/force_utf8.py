"""UTF-8 console enforcement utility."""

from __future__ import annotations

import os
import sys

_WARNED = False


def force_utf8() -> None:
    """Ensure stdout/stderr use UTF-8 and print status once."""

    global _WARNED
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:  # pragma: no cover - depends on runtime
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:  # pragma: no cover
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not _WARNED:
        print("[ENCODING] utf8=on")
        _WARNED = True


__all__ = ["force_utf8"]
