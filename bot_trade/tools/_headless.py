from __future__ import annotations

"""Headless backend helper ensuring single Agg notice per process."""

import matplotlib

_DONE = False


def ensure_headless_once(cli_name: str | None = None) -> None:
    """Force matplotlib backend to Agg and print once."""
    global _DONE
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    if not _DONE:
        backend = matplotlib.get_backend()
        backend = backend[0].upper() + backend[1:] if backend else backend
        print(f"[HEADLESS] backend={backend}")
        _DONE = True
