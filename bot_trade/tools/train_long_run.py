from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from bot_trade.config.rl_paths import RunPaths, reports_dir
from bot_trade.tools.latest import latest_run
from bot_trade.tools._headless import ensure_headless_once


def _summary(rp: RunPaths) -> tuple[float, float]:
    summary_file = rp.performance_dir / "summary.json"
    try:
        data = json.loads(summary_file.read_text())
        return float(data.get("sharpe", 0.0)), float(data.get("max_drawdown", 0.0))
    except Exception:
        return 0.0, 0.0


def main(argv: List[str] | None = None) -> int:
    ensure_headless_once("tools.train_long_run")
    ap = argparse.ArgumentParser(description="Periodic long-run trainer")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--data-dir", dest="data_dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--continuous-env", action="store_true")
    ap.add_argument("--preset", action="append", default=[])
    ap.add_argument("--save-every", type=int, required=True)
    ap.add_argument("--eval-every", type=int, required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--allow-synth", action="store_true")
    ap.add_argument("--ai-core", action="store_true")
    ap.add_argument("--emit-dummy-signals", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tearsheet", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    algo = args.algorithm.upper()
    steps = 0
    algo_root = reports_dir(args.symbol, args.frame, algo).parent.parent
    if args.resume:
        rid = latest_run(args.symbol, args.frame, algo_root)
        if rid:
            rp = RunPaths(args.symbol, args.frame, rid, algo)
            ckpts = sorted(rp.checkpoints_dir.glob("*.zip"))
            if ckpts:
                print(f"[LONGRUN] resume_from={ckpts[-1].stem}")
    while steps < args.max_steps:
        chunk = min(args.save_every, args.max_steps - steps)
        cmd = [
            sys.executable,
            "-m",
            "bot_trade.train_rl",
            "--algorithm",
            algo,
            "--symbol",
            args.symbol,
            "--frame",
            args.frame,
            "--total-steps",
            str(chunk),
            "--checkpoint-every",
            str(args.save_every),
            "--eval-every-steps",
            str(args.eval_every),
            "--device",
            args.device,
        ]
        if args.data_dir:
            cmd.extend(["--data-dir", args.data_dir])
        if args.continuous_env:
            cmd.append("--continuous-env")
        if args.headless:
            cmd.append("--headless")
        if args.allow_synth:
            cmd.append("--allow-synth")
        if args.ai_core:
            cmd.append("--ai-core")
        if args.emit_dummy_signals:
            cmd.append("--emit-dummy-signals")
        if args.dry_run:
            cmd.append("--dry-run")
        for p in args.preset:
            cmd.extend(["--preset", p])
        if args.resume or steps > 0:
            cmd.append("--resume-auto")
        subprocess.run(cmd, check=True)
        eval_cmd = [
            sys.executable,
            "-m",
            "bot_trade.tools.eval_run",
            "--symbol",
            args.symbol,
            "--frame",
            args.frame,
            "--run-id",
            "latest",
            "--algorithm",
            algo,
        ]
        if args.tearsheet:
            eval_cmd.append("--tearsheet")
        subprocess.run(eval_cmd, check=True)
        rid = latest_run(args.symbol, args.frame, algo_root)
        rp = RunPaths(args.symbol, args.frame, rid, algo)
        sharpe, maxdd = _summary(rp)
        steps += chunk
        print(
            f"[LONGRUN] step={steps}/{args.max_steps} eval_sharpe={sharpe:.2f} maxdd={maxdd:.2f} out={rp.performance_dir.resolve()}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
