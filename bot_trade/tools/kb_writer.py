from __future__ import annotations

"""Utility for appending run metadata to the knowledge base.

This module centralizes writes to the Knowlogy knowledge base. Entries are
stored in JSON Lines format with one JSON object per line.  Writes are
performed atomically to avoid corrupting the file.
"""

from pathlib import Path
import json
from typing import Any, Mapping, Optional

from bot_trade.config.rl_paths import DEFAULT_KB_FILE
from bot_trade.tools.atomic_io import append_jsonl
from bot_trade.config.rl_builders import _condense_policy_kwargs


KB_DEFAULTS = {
    "run_id": "",
    "symbol": "",
    "frame": "",
    "algorithm": None,
    "algo_meta": {},
    "ts": None,
    "images": 0,
    "images_list": [],
    "rows_reward": 0,
    "rows_step": 0,
    "rows_train": 0,
    "rows_risk": 0,
    "rows_callbacks": 0,
    "rows_signals": 0,
    "vecnorm_applied": False,
    "vecnorm_snapshot_saved": False,
    "best": False,
    "last": False,
    "best_model_path": "",
    "regime_log_lines": 0,
    "adaptive_log_lines": 0,
    "eval": {
        "win_rate": None,
        "sharpe": None,
        "sortino": None,
        "calmar": None,
        "max_drawdown": None,
        "avg_trade_pnl": None,
        "turnover": None,
        "slippage_proxy": None,
    },
    "portfolio": {
        "equity": None,
        "cash": None,
        "positions": None,
        "step": None,
    },
    "regime": {"active": "", "distribution": {}},
    "notes": "",
}


def _resolve_kb_path(run_paths: Any, kb_file: Optional[str] = None) -> Path:
    """Return KB path derived from ``run_paths`` or explicit override."""

    if kb_file:
        return Path(kb_file)
    if isinstance(run_paths, (str, Path)):
        return Path(run_paths)
    if isinstance(run_paths, Mapping):
        kb = run_paths.get("kb_file")
        if kb:
            return Path(kb)
    kb = getattr(run_paths, "kb_file", None)
    if kb:
        return Path(kb)
    return Path(DEFAULT_KB_FILE)


def kb_append(run_paths: Any, payload: dict, kb_file: Optional[str] = None) -> None:
    """Append ``payload`` as JSON to the knowledge base.

    Writes use an atomic temp+replace strategy to avoid corrupting existing
    data. Each line is newline-terminated to maintain JSON Lines format.
    """

    path = _resolve_kb_path(run_paths, kb_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    algo_meta = payload.get("algo_meta") or {}
    if isinstance(algo_meta, dict) and "policy_kwargs" in algo_meta:
        algo_meta["policy_kwargs"] = _condense_policy_kwargs(algo_meta.get("policy_kwargs") or {})
    entry = {
        **KB_DEFAULTS,
        **{k: v for k, v in payload.items() if k not in {"eval", "portfolio", "algo_meta"}},
        "algo_meta": algo_meta,
    }
    entry["eval"] = {**KB_DEFAULTS["eval"], **payload.get("eval", {})}
    entry["portfolio"] = {**KB_DEFAULTS["portfolio"], **payload.get("portfolio", {})}

    # prevent duplicate run_id appends
    if path.exists():
        try:
            with path.open("rb+") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                if size:
                    fh.seek(-1, 1)
                    if fh.read(1) != b"\n":
                        fh.seek(0, 2)
                        fh.write(b"\n")
                    fh.seek(max(size - 4096, 0))
                    data = fh.read()
                else:
                    data = b""
            last_line = data.splitlines()[-1] if data else b""
            last = json.loads(last_line.decode("utf-8")) if last_line else {}
            if last.get("run_id") == entry.get("run_id"):
                return
        except Exception:
            pass

    append_jsonl(path, entry)
