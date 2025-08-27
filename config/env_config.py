import logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv
import os
import yaml

# Load variables from .env if present
load_dotenv()

CONFIG_PATH = os.getenv("BOT_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f) or {}
except Exception:
    CONFIG = {}


# Live trading disabled by default
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Placeholders for API credentials
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")


def get_config() -> dict:
    """Return loaded YAML configuration."""
    return CONFIG
