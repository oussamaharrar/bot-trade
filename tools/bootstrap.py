"""Project root path bootstrap for standalone scripts.

Importing this module ensures the repository root is on ``sys.path`` so
absolute imports like ``from tools.analytics_common import ...`` work even when
scripts are executed directly via ``python tools/xyz.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
