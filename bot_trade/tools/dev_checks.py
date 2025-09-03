from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from bot_trade.config.rl_paths import (
    DEFAULT_KB_FILE,
    DEFAULT_LOGS_DIR,
    DEFAULT_REPORTS_DIR,
)


FLAGS = ["--algorithm", "--sac-warmstart-from-ppo", "--warmstart-from-ppo"]


def _train_help_ok() -> Tuple[bool, str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "bot_trade.train_rl", "--help"],
            text=True,
        )
    except Exception as e:  # pragma: no cover
        return False, f"train_rl --help failed err={e}"
    missing = [f for f in FLAGS if f not in out]
    if missing:
        return False, f"missing flags: {','.join(missing)}"
    return True, ""
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight repo self-checks")
    parser.parse_args(argv)

    reasons = []
    ok, msg = _train_help_ok()
    if not ok:
        reasons.append(msg)
    logs_root = Path(DEFAULT_LOGS_DIR)
    train_logs = []
    if logs_root.exists():
        train_logs = [p for p in logs_root.rglob("train.log") if p.is_file() and not p.is_symlink()]
    if not train_logs:
        print("[LATEST] none")
        return 2
    latest = None
    lines: list[str] = []
    for log in sorted(train_logs, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            cand = log.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        if any("[POSTRUN]" in ln for ln in cand):
            latest = log
            lines = cand
            break
    if latest is None:
        print("[LATEST] none")
        return 2

    l1 = l2 = l3 = None
    for idx, ln in enumerate(lines, 1):
        if l1 is None and "[DEBUG_EXPORT]" in ln:
            l1 = idx
        if l1 is not None and l2 is None and "[CHARTS]" in ln:
            l2 = idx
        if l2 is not None and l3 is None and "[POSTRUN]" in ln:
            l3 = idx
    if not (l1 and l2 and l3 and l1 < l2 < l3):
        reasons.append("log order")
    post_line = next((ln for ln in lines if "[POSTRUN]" in ln), "")
    m_run = re.search(r"run_id=([^ ]+)", post_line)
    m_algo = re.search(r"algorithm=([^ ]+)", post_line)
    run_id = m_run.group(1) if m_run else None
    algo = m_algo.group(1) if m_algo else None

    if run_id and algo:
        frame = latest.parent.parent.name
        symbol = latest.parent.parent.parent.name
        charts_dir = Path(DEFAULT_REPORTS_DIR) / algo / symbol / frame / run_id / "charts"
        pngs = list(charts_dir.glob("*.png")) if charts_dir.exists() else []
        if not pngs:
            reasons.append("no charts")
        else:
            if not (charts_dir / "risk_flags.png").exists():
                reasons.append("risk_flags.png missing")
            if not (charts_dir / "regimes.png").exists():
                reasons.append("regimes.png missing")
            for p in pngs:
                if p.stat().st_size < 1024:
                    reasons.append(f"small {p.name}")
    else:
        reasons.append("postrun missing fields")

    kb = Path(DEFAULT_KB_FILE)
    if not kb.exists():
        reasons.append("kb missing")
    else:
        try:
            last = json.loads(kb.read_text(encoding="utf-8").splitlines()[-1])
            if run_id and last.get("run_id") != run_id:
                reasons.append("kb run_id mismatch")
            if "algorithm" not in last or last.get("eval", {}).get("max_drawdown") is None:
                reasons.append("kb missing fields")
        except Exception:
            reasons.append("kb invalid")

    if reasons:
        for r in reasons:
            print(f"[DEV_CHECKS] {r}")
        return 1
    print("[DEV_CHECKS] ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
