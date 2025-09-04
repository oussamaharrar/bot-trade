from __future__ import annotations

"""Single headless backend notice per process."""

HEADLESS_MARK = "[HEADLESS] backend=Agg"
_HEADLESS_ONCE = False

def ensure_headless_once() -> None:
    """Set matplotlib backend to Agg and print notice once."""
    global _HEADLESS_ONCE
    if _HEADLESS_ONCE:
        return
    import matplotlib

    matplotlib.use("Agg")
    print(HEADLESS_MARK, flush=True)
    _HEADLESS_ONCE = True
