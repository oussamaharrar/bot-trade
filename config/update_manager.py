"""Centralized update manager for training artefacts.

This module centralizes all disk writes that happen during
reinforcement-learning training.  The intent is to ensure that only the
main process touches the filesystem which avoids broken pipes on Windows
and keeps log formats consistent.

The :class:`UpdateManager` exposes a light‑weight API used by
callbacks/other modules:

``log_step``             – append step level metrics
``update_performance``   – store evaluation statistics
``update_best_model``    – keep track of the best checkpoint
``update_knowledge``     – append structured knowledge events

All information is written under ``results/<SYMBOL>/<FRAME>/logs``.
"""
from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional


def _utcnow() -> str:
    """Return a compact ISO timestamp."""
    return datetime.utcnow().isoformat()


class UpdateManager:
    """Coordinate file writes for training output.

    Parameters
    ----------
    paths: dict
        Mapping produced by :func:`config.rl_paths.build_paths` /
        ``get_paths``.  It must contain at least ``logs_dir`` and
        ``best_zip``.
    symbol, frame: str
        Used only for context in logs and reports.
    cfg: dict, optional
        Optional configuration dictionary (currently unused but kept for
        compatibility with older code).
    """

    def __init__(self, paths: Dict[str, str], symbol: str, frame: str,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        if mp.current_process().name != "MainProcess":
            raise RuntimeError("UpdateManager must run in MainProcess")

        self.paths = paths
        self.symbol = symbol
        self.frame = frame
        self.cfg = cfg or {}

        self.logs_dir = paths.get("logs_dir") or os.path.join(
            paths.get("base", "results"), "logs"
        )
        os.makedirs(self.logs_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Step CSV
        step_path = paths.get("step_csv") or os.path.join(self.logs_dir, "step_log.csv")
        os.makedirs(os.path.dirname(step_path), exist_ok=True)
        self._step_fh = open(step_path, "a", encoding="utf-8", newline="")
        self._step_csv = csv.writer(self._step_fh)
        if os.path.getsize(step_path) == 0:
            self._step_csv.writerow(["ts", "step", "metric", "value"])

        # ------------------------------------------------------------------
        # Performance CSV (aggregated evaluation metrics)
        perf_path = paths.get("perf_csv") or os.path.join(self.logs_dir, "performance.csv")
        os.makedirs(os.path.dirname(perf_path), exist_ok=True)
        self._perf_fh = open(perf_path, "a", encoding="utf-8", newline="")
        self._perf_csv = csv.writer(self._perf_fh)
        if os.path.getsize(perf_path) == 0:
            self._perf_csv.writerow(["ts", "metric", "value"])

        # ------------------------------------------------------------------
        # Knowledge aggregation
        self._knowledge_events = os.path.join(self.logs_dir, "knowledge_events.jsonl")
        self._kb_full = paths.get("kb_file") or os.path.join("memory", "knowledge_base_full.json")

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------
    def log_step(self, step: int, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Append step level metrics to ``step_log.csv``.

        Parameters
        ----------
        step: int
            Current global step count.
        metrics: dict, optional
            Mapping of metric name to value.  If empty, a placeholder row
            is written so that the step file still reflects progress.
        """

        metrics = metrics or {}
        if self._step_csv is None:
            return
        ts = _utcnow()
        if not metrics:
            self._step_csv.writerow([ts, step, "", ""])
        else:
            for key, value in metrics.items():
                self._step_csv.writerow([ts, step, key, value])
        self._step_fh.flush()

    # Compatibility shim (older code used ``on_step``)
    def on_step(self, step: int, metrics: Optional[Dict[str, Any]] = None) -> None:  # pragma: no cover - legacy
        self.log_step(step, metrics)

    def update_performance(self, metrics: Dict[str, Any]) -> None:
        """Record aggregated evaluation statistics.

        Each key/value pair in ``metrics`` is appended to
        ``performance.csv`` along with the current timestamp.
        """

        if not metrics:
            return
        ts = _utcnow()
        for k, v in metrics.items():
            self._perf_csv.writerow([ts, k, v])
        self._perf_fh.flush()

    def on_eval_end(self, eval_metrics: Optional[Dict[str, Any]] = None) -> None:  # pragma: no cover - legacy
        if eval_metrics:
            self.update_performance(eval_metrics)

    def update_best_model(self, src_path: str, metric: Optional[float] = None) -> None:
        """Persist the current best checkpoint.

        Parameters
        ----------
        src_path: str
            Path to the newly evaluated model.
        metric: float, optional
            Performance metric used to rank the model (e.g., mean reward).
        """

        if not src_path:
            return
        dst = self.paths.get("best_zip") or self.paths.get("model_best_zip")
        if not dst:
            return
        if not os.path.exists(src_path):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src_path, dst)
        logging.info("[UpdateManager] saved best model to %s", dst)

        # Record metadata for easy inspection
        meta_path = self.paths.get("best_meta") or os.path.join(os.path.dirname(dst), "best_ckpt.json")
        meta = {"ts": _utcnow(), "path": dst}
        if metric is not None:
            meta["metric"] = float(metric)
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            logging.warning("[UpdateManager] failed to write %s: %s", meta_path, exc)

    def update_knowledge(self, event: Dict[str, Any]) -> None:
        """Append a knowledge event and refresh ``knowledge_base_full.json``.

        The event is appended as JSONL to ``knowledge_events.jsonl`` and
        also merged into the global knowledge base file used by the
        project.  The knowledge base is treated as a simple list of
        events for this minimal implementation.
        """

        if not event:
            return
        os.makedirs(os.path.dirname(self._knowledge_events), exist_ok=True)
        with open(self._knowledge_events, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Aggregate into knowledge_base_full.json
        kb: list[Any]
        try:
            with open(self._kb_full, "r", encoding="utf-8") as fh:
                kb = json.load(fh)
            if not isinstance(kb, list):
                kb = []
        except Exception:
            kb = []
        kb.append(event)
        try:
            with open(self._kb_full, "w", encoding="utf-8") as fh:
                json.dump(kb, fh, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover
            logging.warning("[UpdateManager] failed to update knowledge base: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def on_rollout_end(self, info: Optional[Dict[str, Any]] = None) -> None:  # pragma: no cover - placeholder
        """Hook for end-of-rollout events (currently no-op)."""
        return

    def on_training_end(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Finalize writers and optionally persist the best model."""
        if summary:
            best_path = summary.get("best_model_path")
            metric = summary.get("metric")
            if best_path:
                self.update_best_model(best_path, metric)
        try:
            self._step_fh.flush()
            self._perf_fh.flush()
        finally:
            try:
                self._step_fh.close()
            except Exception:
                pass
            try:
                self._perf_fh.close()
            except Exception:
                pass
