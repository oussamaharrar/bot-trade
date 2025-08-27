import os
import shutil
import json
import csv
import multiprocessing as mp
from typing import Dict, Optional

from .rl_paths import get_paths

class UpdateManager:
    """Central coordinator for log/knowledge/report updates.

    The manager is intentionally lightweight; heavy aggregation is
    deferred to external tooling (e.g. ``tools/knowledge_sync.py``).
    All file writes occur from the main process only which keeps the
    code Windows friendly.
    """

    def __init__(self, paths: Dict[str, str], symbol: str, frame: str, cfg: Optional[Dict] = None) -> None:
        assert mp.current_process().name == "MainProcess", "UpdateManager must run in MainProcess"
        self.paths = paths
        self.symbol = symbol
        self.frame = frame
        self.cfg = cfg or {}
        # ensure required dirs
        for key in ("logs_dir", "report_dir", "perf_dir"):
            d = self.paths.get(key)
            if d:
                os.makedirs(d, exist_ok=True)
        # open csv writers lazily
        self._step_writer = None
        self._step_fp = None

    # ------------------------------------------------------------------
    # helpers
    def _ensure_step_writer(self) -> csv.writer:
        if self._step_writer is None:
            os.makedirs(os.path.dirname(self.paths["step_csv"]), exist_ok=True)
            self._step_fp = open(self.paths["step_csv"], "a", newline="", encoding="utf-8")
            self._step_writer = csv.writer(self._step_fp)
        return self._step_writer

    def close(self) -> None:
        if self._step_fp:
            try:
                self._step_fp.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # SB3 hook proxies
    def on_step(self, step: int, metrics: Optional[Dict] = None) -> None:
        """Record step level metrics to ``step_csv``."""
        writer = self._ensure_step_writer()
        metrics = metrics or {}
        row = [step]
        for k in sorted(metrics.keys()):
            row.append(metrics[k])
        writer.writerow(row)
        if self._step_fp:
            self._step_fp.flush()

    def on_rollout_end(self, info: Optional[Dict] = None) -> None:
        pass  # placeholder for future extensions

    def on_eval_end(self, eval_metrics: Optional[Dict] = None) -> None:
        if not eval_metrics:
            return
        # update best model if requested
        metric = eval_metrics.get("metric")
        improved = eval_metrics.get("improved")
        tmp_path = eval_metrics.get("tmp_path")
        if improved and tmp_path and os.path.exists(tmp_path):
            os.makedirs(os.path.dirname(self.paths["best_zip"]), exist_ok=True)
            shutil.copy2(tmp_path, self.paths["best_zip"])

    def on_training_end(self, summary: Optional[Dict] = None) -> None:
        self.close()
        # optionally generate report using external tool
        if summary and summary.get("report"):
            try:
                from tools.generate_markdown_report import generate_report  # type: ignore
                generate_report(self.symbol, self.frame, self.paths)
            except Exception:
                pass

__all__ = ["UpdateManager", "get_paths"]
