from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from bot_trade.tools.atomic_io import (
    write_png,
    write_html_atomic,
    write_pdf_atomic,
)


def generate_tearsheet(rp, pdf: bool = False) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    import matplotlib.pyplot as plt
    from bot_trade.eval.utils import load_returns, load_trades
    from bot_trade.eval import metrics
    try:
        from weasyprint import HTML as _HTML  # type: ignore
    except Exception:  # pragma: no cover - optional dep
        _HTML = None

    perf = rp.performance_dir
    summary: Dict[str, Any] = {}
    wfa: Dict[str, Any] = {}
    s_path = perf / "summary.json"
    if s_path.exists():
        try:
            summary = json.loads(s_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    wfa_path = perf / "wfa_summary.json"
    if wfa_path.exists():
        try:
            wfa = json.loads(wfa_path.read_text(encoding="utf-8"))
        except Exception:
            wfa = {}
    returns = load_returns(rp.logs)
    rewards_df = returns.to_frame(name="reward")
    eq = metrics.equity_from_rewards(rewards_df)
    dd_series = eq / eq.cummax() - 1 if not eq.empty else pd.Series(dtype=float)
    trades = load_trades(rp.logs)

    figs: Dict[str, str] = {}
    if not eq.empty:
        fig, ax = plt.subplots()
        ax.plot(eq)
        ax.set_title("Equity Curve")
        eq_path = perf / "tearsheet_equity.png"
        write_png(eq_path, fig)
        plt.close(fig)
        figs["equity"] = eq_path.name

        fig, ax = plt.subplots()
        ax.plot(dd_series)
        ax.set_title("Drawdown")
        dd_path = perf / "tearsheet_drawdown.png"
        write_png(dd_path, fig)
        plt.close(fig)
        figs["drawdown"] = dd_path.name

        fig, ax = plt.subplots()
        ax.hist(returns, bins=20)
        ax.set_title("Return Distribution")
        hist_path = perf / "tearsheet_returns.png"
        write_png(hist_path, fig)
        plt.close(fig)
        figs["returns"] = hist_path.name

        if len(returns) > 30:
            roll = returns.rolling(30).apply(lambda x: metrics.sharpe(x), raw=False)
            fig, ax = plt.subplots()
            ax.plot(roll)
            ax.set_title("Rolling Sharpe (30)")
            roll_path = perf / "tearsheet_roll_sharpe.png"
            write_png(roll_path, fig)
            plt.close(fig)
            figs["roll_sharpe"] = roll_path.name

    if not trades.empty and "pnl" in trades.columns:
        wins = (trades["pnl"] > 0).sum()
        losses = (trades["pnl"] <= 0).sum()
        fig, ax = plt.subplots()
        ax.bar(["win", "loss"], [wins, losses])
        ax.set_title("Trades Win/Loss")
        tr_path = perf / "tearsheet_trades.png"
        write_png(tr_path, fig)
        plt.close(fig)
        figs["trades"] = tr_path.name

    risk_counts: Dict[str, int] = {}
    risk_path = rp.logs / "risk_log.csv"
    if risk_path.exists():
        try:
            risk_df = pd.read_csv(risk_path)
            if "reason" in risk_df:
                risk_counts = risk_df["reason"].value_counts().to_dict()
        except Exception:
            risk_counts = {}
    if risk_counts:
        fig, ax = plt.subplots()
        ax.bar(list(risk_counts.keys()), list(risk_counts.values()))
        ax.set_title("Risk Flags")
        rf_path = perf / "tearsheet_risk.png"
        write_png(rf_path, fig)
        plt.close(fig)
        figs["risk"] = rf_path.name

    regime_counts: Dict[str, int] = {}
    reg_path = perf / "adaptive_log.jsonl"
    if reg_path.exists():
        for line in reg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                reg = json.loads(line).get("regime")
                if reg is not None:
                    regime_counts[reg] = regime_counts.get(reg, 0) + 1
            except Exception:
                continue
    if regime_counts:
        fig, ax = plt.subplots()
        ax.bar(list(regime_counts.keys()), list(regime_counts.values()))
        ax.set_title("Regimes")
        rg_path = perf / "tearsheet_regimes.png"
        write_png(rg_path, fig)
        plt.close(fig)
        figs["regimes"] = rg_path.name

    rows = []
    for k, v in summary.items():
        rows.append(f"<tr><td>{k}</td><td>{'' if v is None else v}</td></tr>")
    if wfa.get("aggregate"):
        rows.append("<tr><th colspan=2>WFA Aggregate</th></tr>")
        for k, v in wfa["aggregate"].items():
            rows.append(f"<tr><td>{k}</td><td>{'' if v is None else v}</td></tr>")
    table = (
        "<table>" + "".join(rows) + "</table>" if rows else "<div>NO DATA</div>"
    )
    images_html = []
    for key in [
        "equity",
        "drawdown",
        "returns",
        "roll_sharpe",
        "trades",
        "risk",
        "regimes",
    ]:
        if key in figs:
            images_html.append(f'<img src="{figs[key]}" alt="{key}" />')
        else:
            images_html.append(f"<div>NO DATA: {key}</div>")
    html = "<html><body>" + table + "".join(images_html) + "</body></html>"
    html_path = perf / "tearsheet.html"
    write_html_atomic(html_path, html)
    if pdf and _HTML:
        try:
            pdf_bytes = _HTML(string=html).write_pdf()
            if pdf_bytes and len(pdf_bytes) > 100:
                pdf_path = perf / "tearsheet.pdf"
                write_pdf_atomic(pdf_path, pdf_bytes)
        except Exception:
            pass
    print(f"[TEARSHEET] out={html_path.resolve()}")
    return html_path


def main(argv: list[str] | None = None) -> int:
    from bot_trade.config.rl_paths import RunPaths, DEFAULT_REPORTS_DIR
    from bot_trade.tools.latest import latest_run
    from bot_trade.tools._headless import ensure_headless_once

    ensure_headless_once("eval.tearsheet")
    ap = argparse.ArgumentParser(description="Generate performance tearsheet")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--pdf", action="store_true")
    ns = ap.parse_args(argv)

    run_id = ns.run_id
    if not run_id or str(run_id).lower() in {"latest", "last"}:
        rid = latest_run(ns.symbol, ns.frame, Path(DEFAULT_REPORTS_DIR) / "PPO")
        if not rid:
            print("[LATEST] none")
            return 2
        run_id = rid
    rp = RunPaths(ns.symbol, ns.frame, run_id)
    try:
        generate_tearsheet(rp, pdf=ns.pdf)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
