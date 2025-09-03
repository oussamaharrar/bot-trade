from __future__ import annotations

from bot_trade.tools.force_utf8 import force_utf8
from bot_trade.train_rl import main

if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
