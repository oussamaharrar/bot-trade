from pathlib import Path

from bot_trade.config.rl_paths import (
    agents_dir as _agents_dir,
    get_root,
    logs_dir as _logs_dir,
    memory_dir as _memory_dir,
    reports_dir as _reports_dir,
    results_dir as _results_dir,
)

# Project root resolved via rl_paths
ROOT = get_root()

# Core directories anchored at ROOT
DIR_RESULTS = ROOT / "results"
DIR_AGENTS = ROOT / "agents"
DIR_MEMORY = _memory_dir()
DIR_KNOWLEDGE = DIR_MEMORY / "knowledge"
DIR_REPORT = ROOT / "reports"
DIR_LOGS = ROOT / "logs"


def ensure_dirs(*paths: Path) -> None:
    """Ensure that each of ``paths`` exists."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def results_dir(symbol: str, frame: str) -> Path:
    """Return the results directory for ``symbol``/``frame``."""
    return _results_dir(symbol, frame)


def agents_dir(symbol: str, frame: str) -> Path:
    """Return the agents directory for ``symbol``/``frame``."""
    return _agents_dir(symbol, frame)


def report_dir(symbol: str, frame: str, run_id: str) -> Path:  # noqa: ARG001
    """Return the report directory (run_id ignored for compatibility)."""
    return _reports_dir(symbol, frame)


def logs_dir(symbol: str, frame: str, run_id: str) -> Path:  # noqa: ARG001
    """Return the logs directory under results (run_id ignored)."""
    return _logs_dir(symbol, frame)
