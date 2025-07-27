from dotenv import load_dotenv
import os

# Load variables from .env if present
load_dotenv()

# Live trading disabled by default
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Placeholders for API credentials
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
