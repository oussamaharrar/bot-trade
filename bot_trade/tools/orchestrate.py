"""Simple training orchestrator for multiple symbols/frames."""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import yaml


def run(cmd: list[str]) -> int:
    print("[ORCH]", " ".join(cmd))
    return subprocess.call(cmd)


def main(cfg_path: str = "config/default.yaml") -> int:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    syms = cfg["orchestrator"]["symbols"]
    frs = cfg["orchestrator"]["frames"]
    algs = cfg["train"]["algorithms"]
    steps = str(cfg["train"]["total_steps"])
    for s, f, a in itertools.product(syms, frs, algs):
        cmd = [
            "python",
            "-m",
            "bot_trade.train_rl",
            "--data-mode",
            cfg["data"]["mode"],
            "--raw-dir",
            cfg["data"]["raw_dir"],
            "--data-source",
            cfg["data"]["source"],
            "--algorithm",
            a,
            "--symbol",
            s,
            "--frame",
            f,
            "--total-steps",
            steps,
            "--headless",
        ]
        if cfg["data"]["mode"] == "live":
            cmd += [
                "--exchange",
                cfg["data"].get("exchange", "binance"),
                "--ccxt-symbol",
                cfg["data"].get("ccxt_symbol", "BTC/USDT"),
            ]
        run(cmd)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from bot_trade.tools.force_utf8 import force_utf8

    force_utf8()
    raise SystemExit(main())
