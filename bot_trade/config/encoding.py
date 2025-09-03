from __future__ import annotations

"""UTF-8 enforcement helpers for CLI entrypoints."""

import os
import sys

_WARNED = False


def force_utf8() -> None:
    """Force UTF-8 for stdio and print environment/stream status."""

    global _WARNED
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    env_enc = os.environ.get("PYTHONIOENCODING", "")
    out_enc = getattr(sys.stdout, "encoding", "")
    try:  # pragma: no cover - depends on runtime support
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        out_enc = getattr(sys.stdout, "encoding", out_enc)
    except Exception:  # pragma: no cover - some streams lack reconfigure
        pass
    try:  # pragma: no cover - stderr may lack reconfigure
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not _WARNED:
        print(f"[ENCODING] PYTHONIOENCODING={env_enc} stdout={out_enc}")
        _WARNED = True
