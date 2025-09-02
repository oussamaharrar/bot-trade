from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from bot_trade.tools._headless import ensure_headless_once
from bot_trade.tools.atomic_io import write_text
from bot_trade.config.rl_paths import RunPaths, DEFAULT_KB_FILE


def _parse_param_grid(grid: str) -> Dict[str, List[str]]:
    params: Dict[str, List[str]] = {}
    for part in filter(None, [p.strip() for p in grid.split(";")]):
        if "=" not in part:
            continue
        key, vals = part.split("=", 1)
        params[key.strip()] = [v.strip() for v in vals.split("|") if v.strip()]
    return params


def _cartesian(params: Dict[str, List[str]]):
    if not params:
        yield {}
        return
    keys = list(params.keys())
    for combo in itertools.product(*(params[k] for k in keys)):
        yield {k: v for k, v in zip(keys, combo)}


def _append_csv(path: Path, rows: List[Dict[str, str]], header: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp, path)


def _to_float(v: str | None) -> float:
    try:
        return float(v) if v not in {None, "", "null", "NaN"} else float("-inf")
    except Exception:
        return float("-inf")


def main(argv: List[str] | None = None) -> int:
    ensure_headless_once("tools.sweep")
    ap = argparse.ArgumentParser(description="Simple experiments sweeper")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--algo-grid", type=str, default="PPO")
    ap.add_argument("--param-grid", type=str, default="")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--allow-synth", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--out-dir", required=True)
    ns = ap.parse_args(argv)

    out_dir = Path(ns.out_dir)
    if out_dir.exists() and out_dir.is_symlink():
        print("[SWEEP] out_dir_symlink", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    algos = [a.strip().upper() for a in ns.algo_grid.split(",") if a.strip()]
    param_grid = _parse_param_grid(ns.param_grid)

    rows: List[Dict[str, str]] = []
    header = [
        "run_id",
        "algorithm",
        "symbol",
        "frame",
        "total_steps",
        "n_steps",
        "batch_size",
        "eval_win_rate",
        "eval_sharpe",
        "eval_max_drawdown",
        "avg_trade_pnl",
        "images",
        "reward_lines",
        "step_lines",
        "train_lines",
        "risk_lines",
        "signals_lines",
        "status",
        "reason",
    ]

    total_runs = 0
    for params in _cartesian(param_grid):
        for algo in algos:
            for trial in range(ns.trials):
                total_runs += 1
                n_steps = int(params.get("n_steps", 32))
                batch = int(params.get("batch_size", 32))
                total_steps = int(params.get("total_steps", n_steps * 2))
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
                    str(n_steps),
                    "--batch-size",
                    str(batch),
                    "--total-steps",
                    str(total_steps),
                    "--headless",
                ]
                if ns.allow_synth:
                    cmd.append("--allow-synth")
                if ns.data_dir:
                    cmd.extend(["--data-dir", ns.data_dir])
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, check=False
                    )
                except Exception as e:  # pragma: no cover
                    rows.append(
                        {
                            "run_id": "",
                            "algorithm": algo,
                            "symbol": ns.symbol,
                            "frame": ns.frame,
                            "total_steps": str(total_steps),
                            "n_steps": str(n_steps),
                            "batch_size": str(batch),
                            "eval_win_rate": "",
                            "eval_sharpe": "",
                            "eval_max_drawdown": "",
                            "avg_trade_pnl": "",
                            "images": "",
                            "reward_lines": "",
                            "step_lines": "",
                            "train_lines": "",
                            "risk_lines": "",
                            "signals_lines": "",
                            "status": "fail",
                            "reason": f"exec err {e}",
                        }
                    )
                    _append_csv(out_dir / "summary.csv", rows, header)
                    continue
                if proc.returncode != 0:
                    reason = f"exit={proc.returncode} tail={proc.stderr.strip()[-200:]}"
                    rows.append(
                        {
                            "run_id": "",
                            "algorithm": algo,
                            "symbol": ns.symbol,
                            "frame": ns.frame,
                            "total_steps": str(total_steps),
                            "n_steps": str(n_steps),
                            "batch_size": str(batch),
                            "eval_win_rate": "",
                            "eval_sharpe": "",
                            "eval_max_drawdown": "",
                            "avg_trade_pnl": "",
                            "images": "",
                            "reward_lines": "",
                            "step_lines": "",
                            "train_lines": "",
                            "risk_lines": "",
                            "signals_lines": "",
                            "status": "fail",
                            "reason": reason,
                        }
                    )
                    _append_csv(out_dir / "summary.csv", rows, header)
                    continue
                m = re.findall(r"\[POSTRUN\].*", proc.stdout)
                if not m:
                    rows.append(
                        {
                            "run_id": "",
                            "algorithm": algo,
                            "symbol": ns.symbol,
                            "frame": ns.frame,
                            "total_steps": str(total_steps),
                            "n_steps": str(n_steps),
                            "batch_size": str(batch),
                            "eval_win_rate": "",
                            "eval_sharpe": "",
                            "eval_max_drawdown": "",
                            "avg_trade_pnl": "",
                            "images": "",
                            "reward_lines": "",
                            "step_lines": "",
                            "train_lines": "",
                            "risk_lines": "",
                            "signals_lines": "",
                            "status": "fail",
                            "reason": "no postrun",
                        }
                    )
                    _append_csv(out_dir / "summary.csv", rows, header)
                    continue
                line = m[-1]
                parts = dict(re.findall(r"(\w+)=([^ ]+)", line))
                run_id = parts.get("run_id", "")
                rp = RunPaths(ns.symbol, ns.frame, run_id, algo)
                avg_pnl = ""
                kb_path = Path(DEFAULT_KB_FILE)
                if kb_path.exists():
                    try:
                        with kb_path.open("r", encoding="utf-8") as fh:
                            for ln in fh:
                                obj = json.loads(ln)
                                if obj.get("run_id") == run_id:
                                    avg_pnl = obj.get("eval", {}).get("avg_trade_pnl", "")
                    except Exception:
                        pass
                s_json = rp.performance_dir / "summary.json"
                if s_json.exists():
                    try:
                        meta = json.loads(s_json.read_text(encoding="utf-8"))
                        avg_pnl = meta.get("avg_trade_pnl", avg_pnl)
                    except Exception:
                        pass
                rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": algo,
                        "symbol": ns.symbol,
                        "frame": ns.frame,
                        "total_steps": str(total_steps),
                        "n_steps": parts.get("n_steps", str(n_steps)),
                        "batch_size": parts.get("batch_size", str(batch)),
                        "eval_win_rate": parts.get("eval_win_rate", ""),
                        "eval_sharpe": parts.get("eval_sharpe", ""),
                        "eval_max_drawdown": parts.get("eval_max_drawdown", ""),
                        "avg_trade_pnl": str(avg_pnl),
                        "images": parts.get("images", ""),
                        "reward_lines": parts.get("reward_lines", ""),
                        "step_lines": parts.get("step_lines", ""),
                        "train_lines": parts.get("train_lines", ""),
                        "risk_lines": parts.get("risk_lines", ""),
                        "signals_lines": parts.get("signals_lines", ""),
                        "status": "ok",
                        "reason": "",
                    }
                )
                _append_csv(out_dir / "summary.csv", rows, header)

    # summary.md
    try:
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        ok_rows.sort(key=lambda r: _to_float(r.get("eval_sharpe")), reverse=True)
        lines = ["| run_id | algorithm | eval_sharpe | eval_win_rate | eval_max_drawdown | avg_trade_pnl |", "|---|---|---|---|---|---|"]
        for r in ok_rows:
            lines.append(
                f"| {r['run_id']} | {r['algorithm']} | {r['eval_sharpe']} | {r['eval_win_rate']} | {r['eval_max_drawdown']} | {r['avg_trade_pnl']} |"
            )
        write_text(out_dir / "summary.md", "\n".join(lines) + "\n")
    except Exception:
        pass

    best = "none"
    best_algo = ""
    best_sharpe = ""
    if rows:
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        if ok_rows:
            ok_rows.sort(key=lambda r: _to_float(r.get("eval_sharpe")), reverse=True)
            best = ok_rows[0]["run_id"]
            best_algo = ok_rows[0]["algorithm"]
            best_sharpe = ok_rows[0].get("eval_sharpe", "")
    abs_csv = (out_dir / "summary.csv").resolve()
    print(f"[SWEEP] runs={total_runs} best_run={best} algo={best_algo} sharpe={best_sharpe} out={abs_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
