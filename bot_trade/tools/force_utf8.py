"""UTF-8 console enforcement utility."""

from __future__ import annotations

import os
import sys

_PRINTED = False


def force_utf8() -> None:
    """Ensure stdout/stderr use UTF-8 and print status once."""

    global _PRINTED
    os.environ["PYTHONIOENCODING"] = "UTF-8"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:  # pragma: no cover - depends on runtime
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    if not _PRINTED:
        print("[ENCODING] UTF-8")
        _PRINTED = True


__all__ = ["force_utf8"]
