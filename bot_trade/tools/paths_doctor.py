from __future__ import annotations

"""Verify path consistency for latest run."""

import argparse
import os
from pathlib import Path

from bot_trade.config.rl_paths import DEFAULT_REPORTS_DIR, RunPaths
from bot_trade.config.encoding import force_utf8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check run paths")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--algorithm", default=None)
    ns = ap.parse_args(argv)

    algo = ns.algorithm
    reports_root = Path(DEFAULT_REPORTS_DIR)
    base = reports_root
    if not algo:
        for cand in reports_root.iterdir():
            p = cand / ns.symbol.upper() / ns.frame / ns.run_id
            if p.exists():
                algo = cand.name
                break
    if not algo:
        print("[PATHS] none")
        return 1
    run_id = ns.run_id
    link = reports_root / algo / ns.symbol.upper() / ns.frame / ns.run_id
    if ns.run_id == "latest" and link.is_symlink():
        run_id = os.readlink(link)
    rp = RunPaths(ns.symbol, ns.frame, run_id, algo)
    vec_ok = "OK" if rp.vecnorm.parent == rp.features.base else "MISMATCH"
    features_dir = rp.features.base
    snapshots = rp.agents
    tb = rp.logs / "events"
    print(
        f"[PATHS] vecnorm split={vec_ok} features_dir={features_dir} snapshots={snapshots} tensorboard={tb}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
