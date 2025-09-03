from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

from bot_trade.config.rl_paths import DEFAULT_KB_FILE, RunPaths, reports_dir
from bot_trade.tools._headless import ensure_headless_once
from bot_trade.tools.atomic_io import append_jsonl, write_text
from bot_trade.eval.gate import gate_metrics


def _parse_list(val: str) -> List[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) and v is not None else ""


THR_FILE = Path(__file__).resolve().parents[1] / "config" / "sweep_thresholds.yaml"


def _load_thresholds() -> dict:
    try:
        with THR_FILE.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def main(argv: List[str] | None = None) -> int:
    ensure_headless_once("tools.sweep")
    ap = argparse.ArgumentParser(description="CPU-only experiment sweeper")
    ap.add_argument("--mode", choices=["grid", "random"], default="grid")
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--learning-rate", default="")
    ap.add_argument("--batch-size", default="")
    ap.add_argument("--net-arch-preset", default="")
    ap.add_argument("--continuous-env", action="store_true")
    ap.add_argument("--allow-synth", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--data-dir")
    ns, rest = ap.parse_known_args(argv)

    algo = ns.algorithm.upper()
    base_dir = reports_dir(ns.symbol, ns.frame, algo) / f"sweep_{int(time.time())}"
    base_dir.mkdir(parents=True, exist_ok=True)
    runs_path = base_dir / "runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()

    ranges: Dict[str, List[str]] = {}
    if ns.learning_rate:
        ranges["learning_rate"] = _parse_list(ns.learning_rate)
    if ns.batch_size:
        ranges["batch_size"] = _parse_list(ns.batch_size)
    if ns.net_arch_preset:
        ranges["net_arch_preset"] = _parse_list(ns.net_arch_preset)

    combos: List[Dict[str, str]] = []
    if ns.mode == "grid":
        if ranges:
            import itertools

            keys = list(ranges.keys())
            for vals in itertools.product(*[ranges[k] for k in keys]):
                combos.append({k: v for k, v in zip(keys, vals)})
        else:
            combos = [{}]
    else:  # random
        if ranges:
            for _ in range(ns.n_trials):
                combos.append({k: random.choice(v) for k, v in ranges.items()})
        else:
            combos = [{} for _ in range(ns.n_trials)]

    cfg_thr = _load_thresholds()
    thr_max_turnover = float(cfg_thr.get("max_abs_turnover", float("inf")))
    min_chart_count = int(cfg_thr.get("min_chart_count", 0))
    min_png_size = int(cfg_thr.get("min_png_size_bytes", 0))
    min_equity_len = int(cfg_thr.get("min_equity_len", 0))
    unstable_ok = bool(cfg_thr.get("unstable_metrics", True))

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
            "32",
            "--batch-size",
            str(params.get("batch_size", 64)),
            "--total-steps",
            "128",
        ]
        if "learning_rate" in params:
            cmd.extend(["--learning-rate", str(params["learning_rate"])])
        if "net_arch_preset" in params:
            cmd.extend(["--net-arch-preset", str(params["net_arch_preset"])])
        if ns.continuous_env:
            cmd.append("--continuous-env")
        if ns.headless:
            cmd.append("--headless")
        if ns.allow_synth:
            cmd.append("--allow-synth")
        if ns.data_dir:
            cmd.extend(["--data-dir", ns.data_dir])
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
                "charts_count": "0",
                "gate_pass": "",
                "gate_reasons": "",
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
                "charts_count": "0",
                "gate_pass": "",
                "gate_reasons": "",
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

        metrics = {
            "sharpe": summary.get("sharpe"),
            "sortino": summary.get("sortino"),
            "calmar": summary.get("calmar"),
            "max_drawdown": summary.get("max_drawdown"),
            "turnover": summary.get("turnover"),
            "win_rate": summary.get("win_rate"),
            "avg_trade_pnl": summary.get("avg_trade_pnl"),
        }
        if unstable_ok:
            unstable = []
            for k, v in list(metrics.items()):
                if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                    unstable.append(k)
                    metrics[k] = None
            if unstable:
                print(f"[SWEEP_WARN] unstable_metrics run_id={run_id} fields={','.join(unstable)}")

        turnover_val = metrics.get("turnover")
        if turnover_val is not None and abs(turnover_val) > thr_max_turnover:
            print(f"[SWEEP_WARN] high_turnover run_id={run_id} val={turnover_val:.3f}")

        charts = list(rp.charts_dir.glob("*.png"))
        charts_count = len(charts)
        reg_exists = (rp.charts_dir / "regimes.png").exists()
        if charts_count < min_chart_count or not reg_exists:
            print(f"[SWEEP_WARN] insufficient_artifacts run_id={run_id}")
        for p in charts:
            try:
                if p.stat().st_size < min_png_size:
                    print(f"[SWEEP_WARN] small_png run_id={run_id} file={p.name}")
                    break
            except Exception:
                continue
        if int(summary.get("equity_len", 0)) < min_equity_len:
            print(f"[SWEEP_WARN] short_equity run_id={run_id} len={summary.get('equity_len',0)}")

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

        if ns.gate:
            gate_pass, gate_reasons = gate_metrics(metrics)
        else:
            gate_pass, gate_reasons = True, []
        print(
            f"[SWEEP] trial={run_id} gate_pass={str(gate_pass).lower()}"
        )

        row = {
            "run_id": run_id,
            "algorithm": algo,
            "symbol": ns.symbol,
            "frame": ns.frame,
            "eval_sharpe": _fmt(metrics.get("sharpe")),
            "eval_sortino": _fmt(metrics.get("sortino")),
            "eval_calmar": _fmt(metrics.get("calmar")),
            "eval_max_drawdown": _fmt(metrics.get("max_drawdown")),
            "turnover": _fmt(metrics.get("turnover")),
            "win_rate": _fmt(metrics.get("win_rate")),
            "avg_pnl": _fmt(metrics.get("avg_trade_pnl")),
            "ai_core_signals": str(ai_signals),
            "regimes_png_exists": str(int(reg_exists)),
            "charts_count": str(charts_count),
            "gate_pass": str(gate_pass).lower(),
            "gate_reasons": ",".join(gate_reasons),
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
    pass_rows = [r for r in rows_sorted if str(r.get("gate_pass", "")).lower() == "true"]

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
        "charts_count",
        "gate_pass",
        "gate_reasons",
        "status",
        "reason",
    ]

    csv_path = base_dir / "summary.csv"
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)
    os.replace(tmp, csv_path)

    md_lines = [
        "| run_id | algorithm | eval_sharpe | eval_max_drawdown | win_rate | gate_pass |",
        "|---|---|---|---|---|---|",
    ]
    for r in pass_rows[: min(5, len(pass_rows))]:
        md_lines.append(
            f"| {r['run_id']} | {r['algorithm']} | {r['eval_sharpe']} | {r['eval_max_drawdown']} | {r['win_rate']} | {r['gate_pass']} |"
        )
    write_text(base_dir / "summary.md", "\n".join(md_lines) + "\n")

    completed = sum(1 for r in rows if r["status"] == "ok")
    skipped = len(rows) - completed

    if completed == 0:
        reasons = {r.get("reason", "") for r in rows}
        common = reasons.pop() if len(reasons) == 1 else "mixed"
        print(f"[SWEEP_WARN] all_trials_skipped reason=\"{common}\"")

    top = pass_rows[0] if pass_rows else (rows_sorted[0] if rows_sorted else {})
    top_sharpe = _to_float(top.get("eval_sharpe", "nan"))
    if rows_sorted and not math.isnan(top_sharpe) and top_sharpe < 0.0:
        print(f"[SWEEP_WARN] low_top_sharpe={top.get('eval_sharpe','')}")

    best_sharpe = top.get("eval_sharpe", "") if rows_sorted else ""
    gate_pass_count = sum(1 for r in rows if str(r.get("gate_pass", "")).lower() == "true")
    top_run = top.get("run_id", "")
    print(
        f"[SWEEP] trials={len(rows)} completed={completed} skipped={skipped} mode={ns.mode} top_sharpe={best_sharpe or 'null'} top_run={top_run} gate_pass_count={gate_pass_count} out={base_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

