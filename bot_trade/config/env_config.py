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


def _load_config(path: str | Path | None = None) -> dict:
    """Load configuration from ``path`` with fallbacks."""

    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.getenv("BOT_CONFIG")
    if env:
        candidates.append(Path(env))
    candidates.append(_HERE / "config.yaml")
    candidates.append(_HERE / "config.default.yaml")
    for p in candidates:
        try:
            if p and p.exists():
                with p.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except Exception:
            continue
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
