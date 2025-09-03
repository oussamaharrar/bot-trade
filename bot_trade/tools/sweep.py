from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from bot_trade.tools._headless import ensure_headless_once
from bot_trade.tools.atomic_io import append_jsonl, write_text
from bot_trade.config.rl_paths import RunPaths, reports_dir, DEFAULT_KB_FILE


def _parse_grid(grid: str) -> Dict[str, List[str]]:
    params: Dict[str, List[str]] = {}
    for part in filter(None, [p.strip() for p in grid.split(";")]):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        vals = [x.strip() for x in v.strip().strip("[]").split("|") if x.strip()]
        params[k.strip()] = vals
    return params


def _choose(params: Dict[str, List[str]]) -> Dict[str, str]:
    return {k: random.choice(v) for k, v in params.items()}


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else ""


def main(argv: List[str] | None = None) -> int:
    ensure_headless_once("tools.sweep")
    ap = argparse.ArgumentParser(description="CPU-only experiment sweeper")
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--param-grid", default="")
    ap.add_argument("--continuous-env", action="store_true")
    ap.add_argument("--allow-synth", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--data-dir")
    ns, rest = ap.parse_known_args(argv)

    algo = ns.algorithm.upper()
    base_dir = reports_dir(ns.symbol, ns.frame, algo) / f"sweep_{int(time.time())}"
    base_dir.mkdir(parents=True, exist_ok=True)
    runs_path = base_dir / "runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()

    grid = _parse_grid(ns.param_grid)
    combos: List[Dict[str, str]] = []
    if ns.mode == "grid":
        if not grid:
            combos = [{}]
        else:
            keys = list(grid.keys())
            import itertools

            for vals in itertools.product(*[grid[k] for k in keys]):
                combos.append({k: v for k, v in zip(keys, vals)})
    else:  # random
        for _ in range(ns.n_trials):
            combos.append(_choose(grid) if grid else {})

    rows: List[Dict[str, str]] = []

    for params in combos:
        cmd = [
            sys.executable,
            "-m",
            "bot_trade.train_rl",
            "--algorithm",
            algo,
            "--symbol",
            ns.symbol,
            "--frame",
            ns.frame,
            "--device",
            "cpu",
            "--n-envs",
            "1",
            "--n-steps",
            str(params.get("n_steps", 32)),
            "--batch-size",
            str(params.get("batch_size", 64)),
            "--total-steps",
            str(params.get("total_steps", 128)),
        ]
        if ns.continuous_env:
            cmd.append("--continuous-env")
        if ns.headless:
            cmd.append("--headless")
        if ns.allow_synth:
            cmd.append("--allow-synth")
        if ns.data_dir:
            cmd.extend(["--data-dir", ns.data_dir])
        for k, v in params.items():
            if k in {"n_steps", "batch_size", "total_steps"}:
                continue
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])
        cmd.extend(rest)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout + proc.stderr
        run_id = None
        if proc.returncode == 0:
            import re

            m = re.search(r"run_id=([^ ]+)", stdout)
            if m:
                run_id = m.group(1)
        if proc.returncode != 0 or not run_id:
            reason = "algo_guard" if "[ALGO_GUARD]" in stdout else f"exit={proc.returncode}"
            row = {
                "run_id": run_id or "",
                "algorithm": algo,
                "symbol": ns.symbol,
                "frame": ns.frame,
                "eval_sharpe": "",
                "eval_sortino": "",
                "eval_calmar": "",
                "eval_max_drawdown": "",
                "turnover": "",
                "win_rate": "",
                "avg_pnl": "",
                "ai_core_signals": "",
                "regimes_png_exists": "0",
                "status": "skipped",
                "reason": reason,
            }
            rows.append(row)
            append_jsonl(runs_path, row)
            continue

        eval_cmd = [
            sys.executable,
            "-m",
            "bot_trade.tools.eval_run",
            "--symbol",
            ns.symbol,
            "--frame",
            ns.frame,
            "--run-id",
            run_id,
            "--algorithm",
            algo,
        ]
        eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True)
        if eval_proc.returncode != 0:
            reason = "latest_none" if "[LATEST] none" in eval_proc.stdout else f"exit={eval_proc.returncode}"
            row = {
                "run_id": run_id,
                "algorithm": algo,
                "symbol": ns.symbol,
                "frame": ns.frame,
                "eval_sharpe": "",
                "eval_sortino": "",
                "eval_calmar": "",
                "eval_max_drawdown": "",
                "turnover": "",
                "win_rate": "",
                "avg_pnl": "",
                "ai_core_signals": "",
                "regimes_png_exists": "0",
                "status": "skipped",
                "reason": reason,
            }
            rows.append(row)
            append_jsonl(runs_path, row)
            continue

        rp = RunPaths(ns.symbol, ns.frame, run_id, algo)
        summary = {}
        try:
            summary = json.loads((rp.performance_dir / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        kb_path = Path(DEFAULT_KB_FILE)
        ai_signals = 0
        if kb_path.exists():
            try:
                with kb_path.open("r", encoding="utf-8") as fh:
                    for ln in fh:
                        obj = json.loads(ln)
                        if obj.get("run_id") == run_id:
                            ai_signals = obj.get("ai_core", {}).get("signals_count", 0)
            except Exception:
                ai_signals = 0
        reg_exists = 1 if (rp.charts_dir / "regimes.png").exists() else 0
        row = {
            "run_id": run_id,
            "algorithm": algo,
            "symbol": ns.symbol,
            "frame": ns.frame,
            "eval_sharpe": _fmt(summary.get("sharpe")),
            "eval_sortino": _fmt(summary.get("sortino")),
            "eval_calmar": _fmt(summary.get("calmar")),
            "eval_max_drawdown": _fmt(summary.get("max_drawdown")),
            "turnover": _fmt(summary.get("turnover")),
            "win_rate": _fmt(summary.get("win_rate")),
            "avg_pnl": _fmt(summary.get("avg_trade_pnl")),
            "ai_core_signals": str(ai_signals),
            "regimes_png_exists": str(reg_exists),
            "status": "ok",
            "reason": "",
        }
        rows.append(row)
        append_jsonl(runs_path, row)

    def _to_float(v: str) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -_to_float(r.get("eval_sharpe", "nan")),
            _to_float(r.get("eval_max_drawdown", "nan")),
        ),
    )
    header = [
        "run_id",
        "algorithm",
        "symbol",
        "frame",
        "eval_sharpe",
        "eval_sortino",
        "eval_calmar",
        "eval_max_drawdown",
        "turnover",
        "win_rate",
        "avg_pnl",
        "ai_core_signals",
        "regimes_png_exists",
        "status",
        "reason",
    ]
    import csv

    csv_path = base_dir / "summary.csv"
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)
    os.replace(tmp, csv_path)

    md_lines = [
        "| run_id | algorithm | eval_sharpe | eval_max_drawdown | win_rate |",
        "|---|---|---|---|---|",
    ]
    for r in rows_sorted[: min(5, len(rows_sorted))]:
        md_lines.append(
            f"| {r['run_id']} | {r['algorithm']} | {r['eval_sharpe']} | {r['eval_max_drawdown']} | {r['win_rate']} |"
        )
    write_text(base_dir / "summary.md", "\n".join(md_lines) + "\n")

    best_sharpe = rows_sorted[0]["eval_sharpe"] if rows_sorted else ""
    print(
        f"[SWEEP] trials={len(combos)} mode={ns.mode} top_sharpe={best_sharpe or 'null'} out={csv_path.resolve()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

