from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from bot_trade.config.rl_paths import DEFAULT_LOGS_DIR, DEFAULT_KB_FILE


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


def _latest_postrun() -> Tuple[bool, str, Optional[str], Optional[str]]:
    logs_root = Path(DEFAULT_LOGS_DIR)
    train_logs = []
    if logs_root.exists():
        for algo_dir in logs_root.iterdir():
            if not algo_dir.is_dir() or algo_dir.is_symlink():
                continue
            for sym_dir in algo_dir.iterdir():
                if not sym_dir.is_dir() or sym_dir.is_symlink():
                    continue
                for frame_dir in sym_dir.iterdir():
                    if not frame_dir.is_dir() or frame_dir.is_symlink():
                        continue
                    for run_dir in frame_dir.iterdir():
                        if not run_dir.is_dir() or run_dir.is_symlink():
                            continue
                        log = run_dir / "train.log"
                        if log.exists():
                            train_logs.append(log)
    if not train_logs:
        return False, "no train logs", None, None
    latest = max(train_logs, key=lambda p: p.stat().st_mtime)
    try:
        lines = latest.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:  # pragma: no cover
        return False, f"cannot read {latest}: {e}", None, None
    posts = [ln for ln in lines if "[POSTRUN]" in ln]
    if posts:
        line = posts[-1]
        if "algorithm=" not in line or "eval_max_drawdown=" not in line:
            return False, "postrun missing fields", None, None
        m_run = re.search(r"run_id=([^ ]+)", line)
        m_algo = re.search(r"algorithm=([^ ]+)", line)
        run_id = m_run.group(1) if m_run else None
        algo = m_algo.group(1) if m_algo else None
        return True, "", run_id, algo
    # fallback: derive from KB
    kb = Path(DEFAULT_KB_FILE)
    if not kb.exists():
        return False, "no [POSTRUN] line", None, None
    try:
        last = json.loads(kb.read_text(encoding="utf-8").splitlines()[-1])
    except Exception:
        return False, "no [POSTRUN] line", None, None
    if "algorithm" in last and last.get("eval", {}).get("max_drawdown") is not None:
        return True, "", last.get("run_id"), last.get("algorithm")
    return False, "no [POSTRUN] line", None, None


def _kb_check(run_id: Optional[str]) -> Tuple[bool, str]:
    kb = Path(DEFAULT_KB_FILE)
    if not kb.exists():
        return False, "kb missing"
    try:
        lines = kb.read_text(encoding="utf-8").splitlines()
    except Exception as e:  # pragma: no cover
        return False, f"kb read failed err={e}"
    if not lines:
        return False, "kb empty"
    try:
        last = json.loads(lines[-1])
    except Exception:
        return False, "kb last line invalid"
    if "algorithm" not in last:
        return False, "kb missing algorithm"
    if run_id and last.get("run_id") != run_id:
        return False, "kb run_id mismatch"
    runs = []
    for l in lines:
        l = l.strip()
        if not l:
            continue
        try:
            obj = json.loads(l)
        except Exception:
            continue
        runs.append(obj.get("run_id"))
    if run_id and runs.count(run_id) > 1:
        return False, "duplicate run_id"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight repo self-checks")
    parser.parse_args(argv)

    reasons = []
    ok, msg = _train_help_ok()
    if not ok:
        reasons.append(msg)
    ok, msg, run_id, _ = _latest_postrun()
    if not ok:
        reasons.append(msg)
    ok, msg = _kb_check(run_id if ok else None)
    if not ok:
        reasons.append(msg)
    if reasons:
        for r in reasons:
            print(f"[DEV_CHECKS] {r}")
        return 1
    print("[DEV_CHECKS] ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
