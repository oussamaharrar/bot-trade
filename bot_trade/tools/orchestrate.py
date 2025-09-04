from __future__ import annotations

"""Minimal multi-run orchestrator."""

import argparse
import itertools
import subprocess
from pathlib import Path

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    if yaml is None:
        print("[ORCH] PyYAML not installed")
        return 1
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    orch = cfg.get("orchestrator", {})
    symbols = orch.get("symbols", [])
    frames = orch.get("frames", [])
    algos = cfg.get("train", {}).get("algorithms", [])
    seeds = orch.get("seed_grid", [0])
    summary = []
    for sym, fr, algo, seed in itertools.product(symbols, frames, algos, seeds):
        cmd = [
            "python",
            "-m",
            "bot_trade.train_rl",
            "--symbol",
            sym,
            "--frame",
            fr,
            "--algorithm",
            algo,
            "--seed",
            str(seed),
        ]
        subprocess.run(cmd, check=False)
        summary.append({"symbol": sym, "frame": fr, "algorithm": algo, "seed": seed})
    pd.DataFrame(summary).to_csv("orchestrate_summary.csv", index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

