# ai_core/ai_dashboard.py
# Minimal CLI/Report dashboard over KB + session artifacts
from __future__ import annotations
import os, json
from typing import Dict, Any

KB_PATH = os.getenv("KB_FILE", os.path.join("memory", "knowledge_base_full.json"))
PDF_REPORT = os.path.join("reports", "ai_report.pdf")


def _load_json(path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _ensure_dirs() -> None:
    try:
        os.makedirs(os.path.dirname(PDF_REPORT), exist_ok=True)
    except Exception:
        pass


def cli_summary(kb: Dict[str, Any]) -> None:
    strat = kb.get("strategy_memory", {})
    sigs = kb.get("signals_memory", {})
    print("===== Top Frames by reward_mean =====")
    topf = sorted(((k, v.get("reward_mean", 0.0)) for k, v in strat.items()), key=lambda x: x[1], reverse=True)[:5]
    for k, m in topf:
        print(f"  {k:>6}: reward_mean={m:.4f}")
    print("===== Top Signals by win_rate (min 50) =====")
    tops = sorted(((k, v.get("win_rate", 0.0), v.get("count", 0.0)) for k, v in sigs.items()), key=lambda x: x[1], reverse=True)
    for k, wr, c in tops:
        if float(c) < 50:
            continue
        print(f"  {k:>18}: win_rate={wr:.3f} (n={int(c)})")


def pdf_report(kb: Dict[str, Any], out_path: str = PDF_REPORT) -> bool:
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
    except Exception:
        return False
    _ensure_dirs()
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("AI Dashboard Report", styles["Title"]))
    story.append(Spacer(1, 12))
    strat = kb.get("strategy_memory", {})
    sigs = kb.get("signals_memory", {})
    # Frames
    story.append(Paragraph("Top Frames by reward_mean", styles["Heading2"]))
    topf = sorted(((k, v.get("reward_mean", 0.0)) for k, v in strat.items()), key=lambda x: x[1], reverse=True)[:10]
    for k, m in topf:
        story.append(Paragraph(f"{k}: reward_mean={m:.4f}", styles["Normal"]))
    story.append(Spacer(1, 12))
    # Signals
    story.append(Paragraph("Signals by win_rate (min 50)", styles["Heading2"]))
    tops = sorted(((k, v.get("win_rate", 0.0), v.get("count", 0.0)) for k, v in sigs.items()), key=lambda x: x[1], reverse=True)
    for k, wr, c in tops:
        if float(c) < 50:
            continue
        story.append(Paragraph(f"{k}: win_rate={wr:.3f} (n={int(c)})", styles["Normal"]))
    try:
        doc.build(story)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    kb = _load_json(KB_PATH)
    cli_summary(kb)
    ok = pdf_report(kb)
    print("PDF:", "created" if ok else "skipped (reportlab missing)")
