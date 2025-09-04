"""Minimal HTML tearsheet generator."""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

from bot_trade.tools.atomic_io import write_text


def _metrics_table(metrics: Mapping[str, float]) -> str:
    rows = "".join(
        f"<tr><th>{k}</th><td>{v:.4f}</td></tr>" for k, v in metrics.items()
    )
    return f"<table>{rows}</table>"


def generate_tearsheet(run_dir: Path) -> Path:
    """Create a tiny HTML report under ``run_dir``."""
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    charts_dir = run_dir / "charts"
    metrics: dict[str, float] = {}
    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                row = next(reader, None)
                if row:
                    for k, v in row.items():
                        try:
                            metrics.setdefault(k, float(v))
                        except Exception:
                            continue
        except Exception:
            pass
    images = [p.name for p in charts_dir.glob("*.png")] if charts_dir.exists() else []
    img_html = "".join(f"<img src='charts/{name}' alt='{name}'/>" for name in images)
    html = f"<html><body>{_metrics_table(metrics)}{img_html}</body></html>"
    out = run_dir / "tearsheet.html"
    write_text(out, html)
    try:
        from weasyprint import HTML

        HTML(string=html, base_url=str(run_dir)).write_pdf(run_dir / "tearsheet.pdf")
    except Exception:
        print("PDF unavailable; HTML generated")
    return out


__all__ = ["generate_tearsheet"]
