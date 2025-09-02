from __future__ import annotations

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def load_reward_log(log_dir: Path) -> 'pd.DataFrame':
    import pandas as pd

    reward_path = Path(log_dir) / "reward.log"
    rows = []
    if reward_path.exists():
        for line in reward_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                step = int(parts[0])
                reward = float(parts[-1])
                rows.append({"step": step, "reward": reward})
            except Exception:
                continue
    return pd.DataFrame(rows)


def load_returns(log_dir: Path) -> 'pd.Series':
    import pandas as pd

    df = load_reward_log(log_dir)
    return df["reward"] if not df.empty else pd.Series(dtype=float)


def load_trades(log_dir: Path) -> 'pd.DataFrame':
    import pandas as pd

    trades_path = Path(log_dir) / "trades.csv"
    if trades_path.exists():
        try:
            df = pd.read_csv(trades_path, encoding="utf-8", errors="ignore", on_bad_lines="skip")
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()
