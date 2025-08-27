import os
import csv
import json
import shutil
import logging
import multiprocessing as mp
from datetime import datetime
from typing import Dict, Any, Optional


class UpdateManager:
    """Central coordinator for logging and model updates.

    This object must live in the main process and acts as the single
    gateway for any file IO performed during training.  Worker processes
    should communicate metrics/events back to the main process via SB3
    callbacks which call the methods exposed here.
    """

    def __init__(self, paths: Dict[str, str], symbol: str, frame: str, cfg: Optional[Dict[str, Any]] = None):
        assert mp.current_process().name == "MainProcess", "UpdateManager must run in MainProcess"
        self.paths = paths
        self.symbol = symbol
        self.frame = frame
        self.cfg = cfg or {}

        # ensure directories exist
        for key in ("base", "logs_dir", "report_dir", "perf_dir"):
            p = self.paths.get(key)
            if p:
                os.makedirs(p, exist_ok=True)

        # step level csv
        self._step_path = self.paths.get("step_csv") or self.paths.get("steps_csv")
        self._step_writer = None
        if self._step_path:
            os.makedirs(os.path.dirname(self._step_path) or ".", exist_ok=True)
            new_file = not os.path.exists(self._step_path)
            self._step_writer = open(self._step_path, "a", encoding="utf-8", newline="")
            self._step_csv = csv.writer(self._step_writer)
            if new_file:
                self._step_csv.writerow(["ts", "step", "metric", "value"])

        self._last_eval: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    def on_step(self, step: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Record step metrics to the step CSV."""
        if self._step_writer is None:
            return
        ts = datetime.utcnow().isoformat()
        metrics = metrics or {}
        if not metrics:
            self._step_csv.writerow([ts, step, "", ""])
        else:
            for k, v in metrics.items():
                self._step_csv.writerow([ts, step, k, v])
        self._step_writer.flush()

    def on_rollout_end(self, info: Optional[Dict[str, Any]] = None) -> None:
        """Placeholder for rollout end hooks."""
        # could be used for periodic report generation
        return

    def on_eval_end(self, eval_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Store last evaluation metrics for later use."""
        if eval_metrics:
            self._last_eval = dict(eval_metrics)

    def save_best_model(self, src_path: str, metric: Optional[float] = None) -> None:
        """Copy improved model checkpoint to configured best path."""
        try:
            dst = self.paths.get("best_zip")
            if src_path and dst and os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src_path, dst)
                logging.info("[UpdateManager] saved best model to %s", dst)
                if metric is not None:
                    self._last_eval["best_metric"] = float(metric)
        except Exception as e:  # pragma: no cover
            logging.warning("[UpdateManager] save_best_model failed: %s", e)

    def on_training_end(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Finalize writers and generate report if required."""
        if summary:
            best_path = summary.get("best_model_path")
            metric = summary.get("metric")
            if best_path:
                self.save_best_model(best_path, metric)
        if self._step_writer is not None:
            try:
                self._step_writer.flush()
            except Exception:
                pass
            try:
                self._step_writer.close()
            except Exception:
                pass
