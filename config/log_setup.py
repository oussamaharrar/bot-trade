"""
config/log_setup.py — Central logging setup for the whole system

Goals:
  - Single entrypoint to initialize logging for training/eval/tools.
  - Queue-based logging (thread/process safe) with rotating file handlers.
  - Standardized files per (symbol, frame): train.log, benchmark.log, error.log, risk.log
  - Optional JSONL handlers for structured analytics (decisions/sanity/etc.).
  - Console logging toggle.
  - Helpers to gracefully shutdown the listener at program end.

Usage:
    from .rl_paths import build_paths
    from .log_setup import setup_logging, stop_logging

    paths = build_paths(symbol="BTCUSDT", frame="1m")
    log_objs = setup_logging(paths, console=True, jsonl_extra={
        "decisions": paths["logs"]/"entry_decisions.jsonl",
        "sanity": paths["logs"]/"sanity_checks.jsonl",
    })

    logger = logging.getLogger(__name__)
    logger.info("Hello from unified logger!")

    # ... on exit
    stop_logging(log_objs)

Notes:
  - This module does not enforce other modules to use it, but it exposes
    canonical paths and handlers so the rest of the system can plug in easily.
  - For components that write directly to files (e.g., RiskManager CSV writes),
    we standardize their default location via rl_paths and migrate later.
"""
from __future__ import annotations
import os
import json
import logging
import logging.handlers
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Mapping, Any

# ---------------------------------------------
# Formats
# ---------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------
# Small helpers
# ---------------------------------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _rotating_file_handler(path: Path, level: int, max_mb: int = 50, backups: int = 5) -> logging.Handler:
    _ensure_parent(path)
    h = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_mb * 1024 * 1024, backupCount=backups, encoding="utf-8"
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATEFMT))
    return h

class JSONLHandler(logging.Handler):
    """Simple JSONL file writer as a logging handler.
    Use: logger = logging.getLogger("decisions"); logger.addHandler(JSONLHandler(path))
    """
    def __init__(self, path: Path, level: int = logging.INFO) -> None:
        super().__init__(level)
        _ensure_parent(path)
        self.path = path
        # open in append; do not keep open FD to be fork-safe

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.msg
            if isinstance(msg, Mapping):
                payload: Dict[str, Any] = dict(msg)
            else:
                # try parse JSON strings, else wrap
                if isinstance(msg, str):
                    try:
                        payload = json.loads(msg)
                    except Exception:
                        payload = {"message": msg}
                else:
                    payload = {"message": str(msg)}
            payload.setdefault("ts", datetime.utcnow().isoformat(timespec="seconds") + "Z")
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # never raise from handler
            pass

# ---------------------------------------------
# Public API
# ---------------------------------------------
@dataclass
class LogObjects:
    queue: mp.Queue
    listener: logging.handlers.QueueListener
    files: Dict[str, Path]
    jsonl: Dict[str, Path]


def setup_logging(
    paths: Mapping[str, Path],
    *,
    level: int = logging.INFO,
    console: bool = True,
    jsonl_extra: Optional[Mapping[str, Path]] = None,
    max_mb: int = 50,
    backups: int = 5,
) -> LogObjects:
    """Initialize queue-based logging with rotating files.

    Parameters
    ----------
    paths: mapping with at least keys: "logs", "train_log", "benchmark_log", "error_log".
           See rl_paths.build_paths() for the canonical structure.
    console: add a console StreamHandler.
    jsonl_extra: mapping name->Path for extra JSONL streams (e.g. decisions/sanity).
    """
    logs_dir: Path = Path(paths["logs"]) if not isinstance(paths["logs"], Path) else paths["logs"]

    files = {
        "train": logs_dir / "train.log",
        "benchmark": logs_dir / "benchmark.log",
        "error": logs_dir / "error.log",
        # reserve a standard filename for risk manager logs
        "risk": logs_dir / "risk.log",
    }
    jsonl_paths = dict(jsonl_extra or {})

    # Build file handlers
    handlers = [
        _rotating_file_handler(files["train"], level=level, max_mb=max_mb, backups=backups),
        _rotating_file_handler(files["benchmark"], level=logging.INFO, max_mb=max_mb, backups=backups),
        _rotating_file_handler(files["error"], level=logging.WARNING, max_mb=max_mb, backups=backups),
        _rotating_file_handler(files["risk"], level=logging.INFO, max_mb=max_mb, backups=backups),
    ]

    # Optional console
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATEFMT))
        handlers.append(ch)

    # Optional JSONL streams
    for name, p in jsonl_paths.items():
        handlers.append(JSONLHandler(Path(p), level=logging.INFO))

    # Queue-based setup (safe for multiprocess)
    log_queue: mp.Queue = mp.Queue(-1)
    listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()

    root = logging.getLogger()
    root.setLevel(level)
    # replace all existing handlers with one QueueHandler
    root.handlers[:] = [logging.handlers.QueueHandler(log_queue)]

    # Friendly library loggers baseline levels
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Ensure directory exists for JSONL paths
    for p in jsonl_paths.values():
        _ensure_parent(Path(p))

    return LogObjects(queue=log_queue, listener=listener, files=files, jsonl=jsonl_paths)


def stop_logging(log_objs: Optional[LogObjects]) -> None:
    """Stop listener and flush. Safe to call multiple times."""
    if not log_objs:
        return
    try:
        log_objs.listener.stop()
    except Exception:
        pass


# Convenience helpers ---------------------------------------------------------

def log_device_report(device_str: str = "cpu") -> None:
    """Small utility to log a one-shot device report to the root logger."""
    logger = logging.getLogger("device")
    payload = {"event": "device_report", "device": device_str}
    # Try to enrich with psutil and torch info
    try:
        import psutil  # type: ignore
        p = psutil.Process()
        payload.update({
            "cpu_percent": psutil.cpu_percent(interval=None),
            "mem_mb": round(p.memory_info().rss / (1024**2), 1),
        })
    except Exception:
        pass

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            payload.update({
                "cuda_index": int(idx),
                "cuda_name": torch.cuda.get_device_name(idx),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024**2), 1),
                "cuda_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024**2), 1),
            })
    except Exception:
        pass

    logger.info(payload)


def attach_library_loggers(level: int = logging.WARNING) -> None:
    """Normalize noisy third-party loggers to a sane default."""
    noisy = [
        "stable_baselines3.common.vec_env.subproc_vec_env",
        "gymnasium",
        "ta",
        "urllib3",
        "numexpr",
        "matplotlib",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(level)


# =============================================================================
# ❗️Compatibility wrappers for Train_RL.py
# These are thin adapters that re-use the existing setup_logging/Queue infra.
# =============================================================================

def create_loggers(
    results_dir: str | os.PathLike,
    frame: str,
    symbol: str,
    *,
    level: int = logging.INFO,
    console: bool = True,
) -> tuple[mp.Queue, logging.handlers.QueueListener, logging.Logger]:
    """
    Adapter expected by Train_RL.py.
    Builds the same queue-based logging but returns (queue, listener, root_logger).

    Train_RL.py typically calls:
        log_queue, listener, root_logger = create_loggers(results_dir, frame, symbol)
    """
    # Our unified API expects a paths mapping with at least "logs".
    logs_path = Path(results_dir) / "logs"
    paths = {"logs": logs_path}

    # Optional: add JSONL streams here if desired (kept empty by default).
    jsonl_extra: Dict[str, Path] = {}

    log_objs = setup_logging(
        paths=paths,
        level=level,
        console=console,
        jsonl_extra=jsonl_extra,
    )

    # Friendly banner
    logging.getLogger("boot").info("===== Logging started for %s/%s =====", symbol, frame)

    root_logger = logging.getLogger()
    return log_objs.queue, log_objs.listener, root_logger


def setup_worker_logging(log_queue: mp.Queue, *, level: int = logging.INFO) -> None:
    """
    Adapter expected by Train_RL.py to configure logging inside subprocess workers.

    Attaches a QueueHandler to the root logger so all records flow to the main
    process QueueListener created by create_loggers().
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers to avoid duplicate emissions
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass
    root.addHandler(logging.handlers.QueueHandler(log_queue))


__all__ = [
    "LogObjects",
    "setup_logging",
    "stop_logging",
    "log_device_report",
    "attach_library_loggers",
    # Train_RL compatibility
    "create_loggers",
    "setup_worker_logging",
]
