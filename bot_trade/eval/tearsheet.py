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


def generate_tearsheet(rp) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    import matplotlib.pyplot as plt
    from bot_trade.eval.utils import load_returns
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
    wfa_path = perf / "wfa.json"
    if wfa_path.exists():
        try:
            wfa = json.loads(wfa_path.read_text(encoding="utf-8"))
        except Exception:
            wfa = {}
    returns = load_returns(rp.logs)
    eq = metrics.to_equity_from_returns(returns, start=0.0)
    dd_series = eq / eq.cummax() - 1 if not eq.empty else pd.Series(dtype=float)
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
    for key in ["equity", "drawdown", "returns", "roll_sharpe"]:
        if key in figs:
            images_html.append(f'<img src="{figs[key]}" alt="{key}" />')
        else:
            images_html.append(f"<div>NO DATA: {key}</div>")
    html = "<html><body>" + table + "".join(images_html) + "</body></html>"
    html_path = perf / "tearsheet.html"
    write_html_atomic(html_path, html)
    try:
        if _HTML:
            pdf_bytes = _HTML(string=html).write_pdf()
            if pdf_bytes and len(pdf_bytes) > 100:
                pdf_path = perf / "tearsheet.pdf"
                write_pdf_atomic(pdf_path, pdf_bytes)
    except Exception:
        pass
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
        out = generate_tearsheet(rp)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[TEARSHEET] out={out.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
