"""Chart exporter for training runs.

This module aggregates multiple log sources produced during a training run
and renders a fixed set of charts.  Missing inputs produce labelled
placeholders so downstream tooling always receives the expected images.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

from bot_trade.config.rl_paths import RunPaths, DEFAULT_REPORTS_DIR
from bot_trade.tools.atomic_io import write_png
from bot_trade.tools.latest import latest_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALIAS_MAP: Dict[str, str] = {
    "reward_total": "reward",
    "reward": "reward",
    "global_step": "step",
    "step": "step",
    "ts": "ts",
    "timestamp": "ts",
}


def _read_csv_safe(
    path: Path, expected_cols: list[str] | None = None, alias_map: Dict[str, str] | None = None
) -> "pd.DataFrame":
    """Read ``path`` returning an empty frame on failure.

    ``expected_cols`` ensures missing columns exist.  ``alias_map`` normalises
    column names (e.g. ``reward_total`` -> ``reward``).
    """

    import pandas as pd

    if not path.exists() or path.is_dir():
        return pd.DataFrame(columns=expected_cols or [])
    try:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    except Exception:
        return pd.DataFrame(columns=expected_cols or [])
    if alias_map:
        df.rename(columns={c: alias_map.get(c, alias_map.get(c.lower(), c)) for c in df.columns}, inplace=True)
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype="float64" if col != "ts" else "object")
        df = df[expected_cols]
    return df


def _read_jsonl(path: Path) -> "pd.DataFrame":
    import pandas as pd

    if not path.exists() or path.is_dir():
        return pd.DataFrame()
    records = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _placeholder(path: Path, title: str) -> None:
    """Create a labelled placeholder image."""

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, title, ha="center", va="center")
    ax.set_axis_off()
    write_png(path, fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_run_charts(paths: RunPaths, run_id: str, debug: bool = False) -> Tuple[Path, int, Dict[str, int]]:
    """Export charts for ``run_id`` using ``paths``.

    Returns ``(charts_dir, image_count, rows_dict)``.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    rp = paths if isinstance(paths, RunPaths) else RunPaths(paths.symbol, paths.frame, run_id)

    charts_dir = rp.charts_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    reward_file = rp.results / "reward" / "reward.log"
    step_file = rp.logs / "step_log.csv"
    train_file = rp.logs / "train_log.csv"
    risk_file = rp.logs / "risk_log.csv"
    safety_file = rp.performance_dir / "safety_log.jsonl"
    callbacks_file = rp.logs / "callbacks.jsonl"
    signals_file = rp.logs / "signals.csv"

    reward = _read_csv_safe(reward_file, ["step", "reward", "ts"], ALIAS_MAP)
    step = _read_csv_safe(step_file)
    train = _read_csv_safe(train_file)
    risk = _read_csv_safe(risk_file)
    safety = _read_jsonl(safety_file)
    signals = _read_csv_safe(signals_file)

    callbacks_lines = 0
    if callbacks_file.exists():
        try:
            with callbacks_file.open("r", encoding="utf-8") as fh:
                callbacks_lines = sum(1 for line in fh if line.strip())
        except Exception:
            callbacks_lines = 0

    # numeric coercion (keeping ts columns as object)
    for df in (reward, step, train, risk, signals):
        for col in df.columns:
            if col != "ts":
                df[col] = pd.to_numeric(df[col], errors="coerce")
    rows_safety = len(safety)

    for df in (reward, step, train):
        if "step" in df.columns:
            df.dropna(subset=["step"], inplace=True)

    rows_reward = len(reward)
    rows_step = len(step)
    rows_train = len(train)
    rows_risk = len(risk)
    rows_signals = len(signals)

    images = 0

    def save_fig(fig: plt.Figure, name: str) -> None:
        nonlocal images
        path = charts_dir / name
        write_png(path, fig)
        images += 1

    # reward chart
    if rows_reward > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(reward["step"], reward["reward"].fillna(0))
        ax.set_title("reward")
        save_fig(fig, "reward.png")
    else:
        _placeholder(charts_dir / "reward.png", "NO DATA")
        images += 1

    # sharpe ratio chart
    sharpe_series = pd.Series(dtype=float)
    if rows_reward > 1:
        returns = reward["reward"].diff().fillna(0)
        window = min(256, max(1, len(returns)))
        sharpe_series = (returns.rolling(window).mean() / returns.rolling(window).std()).fillna(0)
    if not sharpe_series.dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(reward["step"], sharpe_series)
        ax.set_title("sharpe")
        save_fig(fig, "sharpe.png")
    else:
        _placeholder(charts_dir / "sharpe.png", "NO DATA")
        images += 1

    # loss / entropy from train log
    if rows_train > 0 and {"step", "metric", "value"}.issubset(train.columns):
        loss = train[train["metric"] == "loss"]
        if not loss.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(loss["step"], loss["value"].fillna(0))
            ax.set_title("loss")
            save_fig(fig, "loss.png")
        else:
            _placeholder(charts_dir / "loss.png", "NO DATA")
            images += 1

        ent = train[train["metric"] == "entropy"]
        if not ent.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ent["step"], ent["value"].fillna(0))
            ax.set_title("entropy")
            save_fig(fig, "entropy.png")
        else:
            _placeholder(charts_dir / "entropy.png", "NO DATA")
            images += 1
    else:
        _placeholder(charts_dir / "loss.png", "NO DATA")
        _placeholder(charts_dir / "entropy.png", "NO DATA")
        images += 2

    # risk flags chart
    if rows_risk > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(rows_risk), [1] * rows_risk)
        ax.set_title("risk flags")
        save_fig(fig, "risk_flags.png")
    else:
        _placeholder(charts_dir / "risk_flags.png", "NO DATA")
        images += 1

    # safety flags chart
    if rows_safety > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(range(rows_safety), [1] * rows_safety)
        ax.set_title("safety flags")
        save_fig(fig, "safety.png")
    else:
        _placeholder(charts_dir / "safety.png", "NO DATA")
        images += 1

    # regimes distribution chart
    reg_file = rp.performance_dir / "adaptive_log.jsonl"
    dist: Dict[str, float] = {}
    if reg_file.exists():
        try:
            import json
            from collections import Counter

            counts: Counter[str] = Counter()
            with reg_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        counts[str(rec.get("regime", "unknown"))] += 1
                    except Exception:
                        continue
            total = sum(counts.values())
            if total > 0:
                dist = {k: v / total for k, v in counts.items()}
        except Exception:
            dist = {}
    if dist:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(dist.keys()), list(dist.values()))
        ax.set_title("regimes")
        save_fig(fig, "regimes.png")
    else:
        _placeholder(charts_dir / "regimes.png", "NO DATA")
        images += 1

    images = len([p for p in charts_dir.glob("*.png") if p.is_file() and not p.is_symlink()])

    rows = {
        "reward": rows_reward,
        "step": rows_step,
        "train": rows_train,
        "risk": rows_risk,
        "signals": rows_signals,
        "callbacks": callbacks_lines,
        "safety": rows_safety,
    }

    return charts_dir, images, rows


# ---------------------------------------------------------------------------
# Backwards compatibility wrappers
# ---------------------------------------------------------------------------


def export_for_run(run_paths: RunPaths, debug: bool = False) -> Dict[str, Any]:
    charts_dir, images, rows = export_run_charts(run_paths, run_paths.run_id, debug)
    return {"charts_dir": str(charts_dir), "images": images, "rows": rows}


def export_run(paths: RunPaths, debug: bool = False):  # pragma: no cover
    cd, imgs, rows = export_run_charts(paths, paths.run_id, debug)
    return cd, imgs, rows.get("reward", 0), rows.get("step", 0)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI helper
    import argparse
    from bot_trade.config.rl_paths import get_root

    def _set_headless() -> None:
        import matplotlib

        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
        print("[HEADLESS] backend=Agg")

    _set_headless()

    ap = argparse.ArgumentParser(
        description="Export charts for a training run",
        epilog="Example: python -m bot_trade.tools.export_charts --symbol BTCUSDT --frame 1m --run-id abc123",
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frame", default="1m")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base", default=None, help="Project root directory")
    ap.add_argument("--debug-export", action="store_true")
    ns = ap.parse_args(argv)

    root = Path(ns.base) if ns.base else get_root()
    rid = ns.run_id
    if str(rid).lower() in {"latest", "last"}:
        reports_root = (Path(ns.base).resolve() / "reports" if ns.base else Path(DEFAULT_REPORTS_DIR))
        rid = latest_run(ns.symbol, ns.frame, reports_root / "PPO")
        if not rid:
            print("[LATEST] none")
            return 2
    rp = RunPaths(ns.symbol, ns.frame, str(rid), root=root)
    charts_dir, images, rows = export_run_charts(rp, str(rid), debug=ns.debug_export)
    print(
        "[DEBUG_EXPORT] reward_rows=%d step_rows=%d train_rows=%d risk_rows=%d callbacks_rows=%d signals_rows=%d"
        % (
            rows.get("reward", 0),
            rows.get("step", 0),
            rows.get("train", 0),
            rows.get("risk", 0),
            rows.get("callbacks", 0),
            rows.get("signals", 0),
        )
    )
    print(f"[CHARTS] dir={charts_dir.resolve()} images={images}")
    rg = charts_dir / "regimes.png"
    if rg.exists():
        print(f"[ADAPT_CHARTS] regimes_png={rg.resolve()}")
    return 0 if images > 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

