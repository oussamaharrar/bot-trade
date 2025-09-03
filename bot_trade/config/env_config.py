import logging
import os
from pathlib import Path
import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False

# Load variables from .env if present
load_dotenv()

_HERE = Path(__file__).resolve().parent
_PRINTED = False


def _load_config(path: str | Path | None = None) -> dict:
    """Load configuration from ``path`` with fallbacks."""

    global _PRINTED
    candidates: list[tuple[str, Path]] = []
    if path:
        candidates.append(("cli", Path(path)))
    env = os.getenv("BOT_CONFIG")
    if env:
        candidates.append(("env", Path(env)))
    candidates.append(("user", _HERE / "config.yaml"))
    candidates.append(("default", _HERE / "config.default.yaml"))
    for source, p in candidates:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if not _PRINTED:
                    print(f"[CONFIG] source={source} path={p}")
                    _PRINTED = True
                return data
        except Exception:
            continue
    if not _PRINTED:
        print("[CONFIG] source=none path=None")
        _PRINTED = True
    return {}


CONFIG = _load_config()


# Live trading disabled by default
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Placeholders for API credentials
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")


def get_config(path: str | None = None) -> dict:
    """Return loaded YAML configuration (with optional override)."""

    return _load_config(path) if path else CONFIG
