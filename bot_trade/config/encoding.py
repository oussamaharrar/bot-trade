from __future__ import annotations

"""UTF-8 enforcement helpers for CLI entrypoints."""

import locale
import os
import sys

_WARNED = False


def force_utf8() -> None:
    """Force UTF-8 for stdio on Windows and warn if cp1252 was active."""
    global _WARNED
    enc = (locale.getpreferredencoding(False) or "").lower()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not _WARNED and "cp1252" in enc:
        print("[ENCODING] forced UTF-8")
        _WARNED = True
