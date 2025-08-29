from pathlib import Path

# Project root resolved from this file location
ROOT = Path(__file__).resolve().parents[1]

# Core directories
DIR_RESULTS = ROOT / "results"
DIR_AGENTS = ROOT / "agents"
DIR_MEMORY = ROOT / "memory"
DIR_KNOWLEDGE = DIR_MEMORY / "knowledge"
DIR_REPORT = ROOT / "report"  # NOTE: interface changed here - renamed from export to report
DIR_LOGS = ROOT / "logs"


def ensure_dirs(*paths: Path) -> None:
    """Ensure that each of ``paths`` exists."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def results_dir(symbol: str, frame: str) -> Path:
    """Return the legacy results directory for ``symbol``/``frame``."""
    p = DIR_RESULTS / symbol / frame
    ensure_dirs(p)
    return p


def agents_dir(symbol: str, frame: str) -> Path:
    """Return the agents directory for ``symbol``/``frame``."""
    p = DIR_AGENTS / symbol / frame
    ensure_dirs(p)
    return p


def report_dir(symbol: str, frame: str, run_id: str) -> Path:
    """Return the run specific report directory."""
    p = DIR_REPORT / symbol / frame / run_id
    ensure_dirs(p)
    return p


def logs_dir(symbol: str, frame: str, run_id: str) -> Path:
    """Return the run specific logs directory."""
    p = DIR_LOGS / symbol / frame / run_id
    ensure_dirs(p)
    return p
