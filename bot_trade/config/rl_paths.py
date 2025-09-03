"""Centralized path helpers for bot_trade.

This module exposes a small set of helpers that resolve all project
paths relative to a single project root.  The root is determined once
and cached via :func:`get_root` using the following strategy:

1. If the environment variable ``BOT_TRADE_ROOT`` is set, it is used.
2. Otherwise we walk upwards from the current working directory and
   from the location of this file until either ``pyproject.toml`` or
   ``.git`` is found.

All memory/knowledge artifacts live under ``<ROOT>/memory``.  A legacy
``bot_trade/memory`` directory is migrated automatically with a one time
warning when :func:`memory_dir` is first accessed.
"""

from __future__ import annotations

import os
import shutil
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Dict, Any
import datetime as dt
import json
import uuid


# ---------------------------------------------------------------------------
# root detection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_root() -> Path:
    """Return the project root directory.

    The value is cached so repeated calls are cheap.
    """

    env = os.environ.get("BOT_TRADE_ROOT")
    if env:
        return Path(env).resolve()

    markers = ("pyproject.toml", ".git")

    def _search(start: Path) -> Path | None:
        p = start
        while True:
            if any((p / m).exists() for m in markers):
                return p
            if p.parent == p:
                return None
            p = p.parent

    for candidate in (Path.cwd(), Path(__file__).resolve()):
        found = _search(candidate)
        if found:
            return found

    raise RuntimeError("Could not determine project root")


# ---------------------------------------------------------------------------
# directory helpers
# ---------------------------------------------------------------------------

_legacy_warned = False


def _migrate_legacy_memory(dst: Path) -> None:
    """Move ``bot_trade/memory`` to the new root memory directory."""

    global _legacy_warned
    if _legacy_warned:
        return

    legacy = get_root() / "bot_trade" / "memory"
    if legacy.exists():
        print(
            "[WARNING] legacy bot_trade/memory detected - migrating to <ROOT>/memory",
            file=sys.stderr,
        )
        dst.mkdir(parents=True, exist_ok=True)
        for item in legacy.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.move(str(item), target)
            else:
                os.replace(item, target)
        try:
            legacy.rmdir()
        except OSError:
            pass

    _legacy_warned = True


def memory_dir() -> Path:
    """Return ``<ROOT>/memory`` ensuring it exists and migrate legacy data."""

    d = get_root() / "memory"
    d.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_memory(d)
    return d


def algo_scoped(root: Path, algo: str, symbol: str, frame: str, run_id: str) -> Path:
    """Return a path scoped by algorithm/symbol/frame/run_id."""
    return Path(root) / algo.upper() / symbol.upper() / str(frame) / str(run_id)


def agents_dir(symbol: str, frame: str, algo: str | None = None, run_id: str | None = None) -> Path:
    base = get_root() / "agents"
    if algo:
        base = base / algo.upper()
    base = base / symbol.upper() / str(frame)
    if run_id:
        base = base / run_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _legacy_agents_dir(symbol: str, frame: str, run_id: str | None = None) -> Path:
    base = get_root() / "agents" / symbol.upper() / str(frame)
    if run_id:
        base = base / run_id
    return base

def best_agent(symbol: str, frame: str, algo: str | None = None, run_id: str | None = None) -> Path:
    """Return path to the best agent checkpoint with legacy fallback."""

    path = agents_dir(symbol, frame, algo, run_id) / "deep_rl_best.zip"
    if not path.exists():
        legacy = _legacy_agents_dir(symbol, frame, run_id) / "deep_rl_best.zip"
        if legacy.exists():
            return legacy
    return path


def last_agent(symbol: str, frame: str, algo: str | None = None, run_id: str | None = None) -> Path:
    """Return path to the last agent checkpoint with legacy fallback."""

    path = agents_dir(symbol, frame, algo, run_id) / "deep_rl_last.zip"
    if not path.exists():
        legacy_dir = _legacy_agents_dir(symbol, frame, run_id)
        legacy = legacy_dir / "deep_rl_last.zip"
        if legacy.exists():
            return legacy
        legacy_alt = legacy_dir / "deep_rl.zip"
        if legacy_alt.exists():
            return legacy_alt
    return path


def latest_agent(symbol: str, frame: str, algo: str | None = None, run_id: str | None = None) -> Path:
    """Alias for :func:`last_agent` kept for backward compatibility."""

    return last_agent(symbol, frame, algo, run_id)


_VECNORM_WARNED = False


def vecnorm_path(symbol: str, frame: str, algo: str | None = None, run_id: str | None = None) -> Path:
    """Return path to VecNormalize statistics with legacy shim.

    When ``algo`` and ``run_id`` are provided the canonical path under
    ``RunPaths.features.vecnorm`` is returned.  If the canonical file does not
    exist but a legacy ``results/<SYMBOL>/<FRAME>/vecnorm.pkl`` does, the file
    is migrated and the canonical path returned.  Calls without ``algo``/``run_id``
    fall back to the legacy location and emit a deprecation warning once.
    """

    global _VECNORM_WARNED
    if algo and run_id:
        rp = RunPaths(symbol, frame, run_id, algo)
        path = rp.features.vecnorm
        legacy = _legacy_results_dir(symbol, frame) / "vecnorm.pkl"
        if not path.exists() and legacy.exists():
            if not _VECNORM_WARNED:
                print("[DEPRECATION] migrating vecnorm.pkl to RunPaths.features.vecnorm")
                _VECNORM_WARNED = True
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy), path)
            except Exception:
                return legacy
        return path
    if not _VECNORM_WARNED:
        print("[DEPRECATION] vecnorm_path legacy lookup; use RunPaths.features.vecnorm")
        _VECNORM_WARNED = True
    legacy_dir = _legacy_results_dir(symbol, frame)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    return legacy_dir / "vecnorm.pkl"


def _legacy_results_dir(symbol: str, frame: str) -> Path:
    return get_root() / "results" / symbol.upper() / str(frame)


def results_dir(symbol: str, frame: str, algo: str | None = None) -> Path:
    base = get_root() / "results"
    if algo:
        candidate = base / algo.upper() / symbol.upper() / str(frame)
        if not candidate.exists():
            legacy = _legacy_results_dir(symbol, frame)
            if legacy.exists():
                return legacy
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    d = base / symbol.upper() / str(frame)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _legacy_reports_dir(symbol: str, frame: str) -> Path:
    return get_root() / "reports" / symbol.upper() / str(frame)


def reports_dir(symbol: str, frame: str, algo: str | None = None) -> Path:
    base = get_root() / "reports"
    if algo:
        candidate = base / algo.upper() / symbol.upper() / str(frame)
        if not candidate.exists():
            legacy = _legacy_reports_dir(symbol, frame)
            if legacy.exists():
                return legacy
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    d = base / symbol.upper() / str(frame)
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir(symbol: str, frame: str, run_base: str | None = None, algo: str | None = None) -> Path:
    root = Path(DEFAULT_LOGS_DIR)
    if algo:
        root = root / algo.upper()
    root = root / symbol.upper() / str(frame)
    if run_base:
        root = root / run_base
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else get_root() / p


# ---------------------------------------------------------------------------
# RunPaths dataclass
# ---------------------------------------------------------------------------


class RunPaths:
    """Central path helper for a single training run.

    Directories are structured as::

        logs/<ALGO>/<SYMBOL>/<FRAME>/<RUN_ID>/
        results/<ALGO>/<SYMBOL>/<FRAME>/<RUN_ID>/
        reports/<ALGO>/<SYMBOL>/<FRAME>/<RUN_ID>/

    ``RUN_ID`` is expected to be a short identifier (no timestamps).
    """

    def __init__(
        self,
        symbol: str,
        frame: str,
        run_id: str,
        algo: str = "PPO",
        root: Path | None = None,
        kb_file: str | Path | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.frame = str(frame)
        self.run_id = run_id
        self.algo = algo.upper()
        self.root = root or get_root()
        self.kb_file = Path(kb_file) if kb_file else Path(DEFAULT_KB_FILE)
        self.kb_file.parent.mkdir(parents=True, exist_ok=True)

        self.logs = Path(DEFAULT_LOGS_DIR) / self.algo / self.symbol / self.frame / self.run_id
        self.results = algo_scoped(Path(DEFAULT_RESULTS_DIR), self.algo, self.symbol, self.frame, self.run_id)
        self.reports = algo_scoped(Path(DEFAULT_REPORTS_DIR), self.algo, self.symbol, self.frame, self.run_id)
        self.agents = algo_scoped(Path(DEFAULT_AGENTS_DIR), self.algo, self.symbol, self.frame, self.run_id)
        features_dir = self.reports / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        self.features = type("Features", (), {
            "__init__": lambda self, base: setattr(self, "base", base),
            "vecnorm": property(lambda self: self.base / "vecnorm.pkl"),
        })(features_dir)

        for d in (self.logs, self.results, self.reports):
            d.mkdir(parents=True, exist_ok=True)
            latest = d.parent / "latest"
            try:
                if latest.exists() or latest.is_symlink():
                    if latest.is_dir() and not latest.is_symlink():
                        shutil.rmtree(latest)
                    else:
                        latest.unlink()
                os.symlink(self.run_id, latest)
            except Exception:
                try:
                    if latest.exists() and latest.is_dir():
                        shutil.rmtree(latest)
                    shutil.copytree(d, latest)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def vecnorm_path(self) -> Path:
        return self.features.vecnorm

    @property
    def vecnorm(self) -> Path:
        return self.vecnorm_path

    @property
    def vecnorm_best(self) -> Path:
        return self.agents / "vecnorm_best.pkl"

    @property
    def vecnorm_last(self) -> Path:
        return self.vecnorm

    @property
    def best_model(self) -> Path:
        return self.agents / "deep_rl_best.zip"

    @property
    def last_model(self) -> Path:
        return self.agents / "deep_rl_last.zip"

    @property
    def archive_dir(self) -> Path:
        return self.agents / "archive"

    @property
    def archive_best_dir(self) -> Path:
        return self.agents / "archive_best"

    @property
    def best_meta_path(self) -> Path:
        return self.agents / "best_meta.json"

    @property
    def last_meta_path(self) -> Path:
        return self.agents / "last_meta.json"

    @property
    def charts_dir(self) -> Path:
        d = self.reports / "charts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def performance_dir(self) -> Path:
        d = self.reports / "performance"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def summary_csv_path(self) -> Path:
        return self.reports / "summary.csv"

    @property
    def summary_json_path(self) -> Path:
        return self.reports / "summary.json"

    def ensure(self) -> None:
        for d in (self.agents, self.archive_dir, self.archive_best_dir):
            d.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, str]:
        """Return mapping used by writers/callbacks."""

        def _p(base: Path, name: str) -> str:
            return str(base / name)

        reward_dir = self.results / "reward"
        reward_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {
            "run_id": self.run_id,
            "logs": str(self.logs),
            "logs_dir": str(self.logs),
            "results": str(self.results),
            "base": str(self.results),
            "reports": str(self.reports),
            "agents": str(self.agents),
            "agents_root": str(self.agents),
            "best_model": str(self.best_model),
            "last_model": str(self.last_model),
            "archive_dir": str(self.archive_dir),
            "archive_best_dir": str(self.archive_best_dir),
            "train_csv": _p(self.logs, "train_log.csv"),
            "benchmark_log": _p(self.logs, "benchmark.log"),
            "risk_log": _p(self.logs, "risk_log.csv"),
            "signals_log": _p(self.logs, "signals_log.csv"),
            "callbacks_log": _p(self.logs, "callbacks_log.csv"),
            "step_csv": _p(self.logs, "step_log.csv"),
            "reward_csv": _p(reward_dir, "reward.log"),
            "trade_csv": _p(self.logs, "deep_rl_trades.csv"),
            "eval_csv": _p(self.logs, "evaluation.csv"),
            "perf_csv": _p(self.logs, "performance.csv"),
            "tb_dir": _p(self.logs, "events"),
            "model_zip": str(self.last_model),
            "model_best_zip": str(self.best_model),
            "best_meta": str(self.best_meta_path),
            "last_meta": str(self.last_meta_path),
            "vecnorm_pkl": str(self.vecnorm_path),
            "vecnorm": str(self.vecnorm),
            "vecnorm_best": str(self.vecnorm_best),
            "vecnorm_last": str(self.vecnorm_last),
            "kb_file": str(self.kb_file),
            "charts_dir": _p(self.reports, "charts"),
            "performance_dir": _p(self.reports, "performance"),
            "summary_csv": _p(self.reports, "summary.csv"),
            "summary_json": _p(self.reports, "summary.json"),
        }
        return paths

    def write_run_meta(self, extra: Dict[str, Any] | None = None) -> None:
        meta = {
            "schema": 1,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "frame": self.frame,
            "ts": dt.datetime.utcnow().isoformat(),
            "base": str(self.root),
        }
        if extra:
            meta.update(extra)
        for d in (self.logs, self.results, self.reports):
            tmp = d / "run.json.tmp"
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, d / "run.json")


def new_run_id() -> str:
    """Return a short unique run identifier."""

    return uuid.uuid4().hex[:8]

REQUIRED_KEYS = [
    "agents_root",
    "best_model",
    "last_model",
    "archive_dir",
    "archive_best_dir",
    "logs",
    "results",
    "reports",
    "vecnorm",
    "vecnorm_best",
    "best_meta",
    "last_meta",
    "reward_csv",
    "step_csv",
    "train_csv",
]


def ensure_contract(paths: Dict[str, str]) -> None:
    """Assert that all required run path keys are present."""

    missing = [k for k in REQUIRED_KEYS if k not in paths or paths[k] is None]
    assert not missing, f"[PATHS] missing keys: {missing}"


# ---------------------------------------------------------------------------
# file helpers
# ---------------------------------------------------------------------------

@contextmanager
def ensure_utf8(path: Path | str, csv_newline: bool = True) -> Iterator[object]:
    """Open ``path`` for writing in UTF-8 ensuring its directory exists."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    newline = "" if csv_newline else None
    with p.open("w", encoding="utf-8", newline=newline) as fh:
        yield fh


def _atomic_replace(src: str | Path, dst: str | Path) -> None:
    """Atomically replace ``dst`` with ``src`` via a ``.tmp`` copy."""

    src_p, dst_p = Path(src), Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_name(dst_p.name + ".tmp")
    shutil.copy2(src_p, tmp)
    os.replace(tmp, dst_p)


def stamp_name(stem: str, run_id: str, ts: str, ext: str) -> str:
    """Return run-aware archive filename."""

    return f"{stem}-{run_id}-{ts}{ext}"


# Default paths -------------------------------------------------------------

ROOT = get_root()
DEFAULT_AGENTS_DIR = os.environ.get("BOT_AGENTS_DIR", str(ROOT / "agents"))
DEFAULT_RESULTS_DIR = os.environ.get("BOT_RESULTS_DIR", str(ROOT / "results"))
DEFAULT_REPORTS_DIR = str(Path(os.environ.get("BOT_REPORTS_DIR", ROOT / "reports")).resolve())
DEFAULT_LOGS_DIR = os.environ.get("BOT_LOGS_DIR", str(ROOT / "logs"))
DEFAULT_MEMORY_FILE = os.environ.get(
    "BOT_MEMORY_FILE", str(memory_dir() / "memory.json")
)
DEFAULT_KB_FILE = os.environ.get(
    "BOT_KB_FILE", str(memory_dir() / "Knowlogy" / "kb.jsonl")
)


# ---------------------------------------------------------------------------
# legacy helpers retained for compatibility
# ---------------------------------------------------------------------------

def _mk(*parts: str) -> str:
    p = os.path.join(*parts)
    os.makedirs(p, exist_ok=True)
    return p


def build_paths(
    symbol: str,
    frame: str,
    run_id: str | None = None,
    *,
    agents_dir: str | None = None,
    results_dir: str | None = None,
    reports_dir: str | None = None,
    logs_dir: str | None = None,
) -> dict:
    """Return run-aware directory map for training artefacts.

    Parameters
    ----------
    run_id: str | None
        Unique identifier for this run.  If ``None`` a placeholder ``dev`` is
        used.  A timestamp suffix is added so multiple runs with the same
        identifier do not collide.
    """

    agents_dir = agents_dir or DEFAULT_AGENTS_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    reports_dir = reports_dir or DEFAULT_REPORTS_DIR
    logs_root = logs_dir or DEFAULT_LOGS_DIR

    sym, frm = symbol.upper(), str(frame)
    rid = run_id or "dev"
    run_ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_base = f"rl-{rid}-{run_ts}"

    paths: dict[str, str] = {
        "run_id": rid,
        "run_ts": run_ts,
        "run_base": run_base,
    }

    paths["agents"] = _mk(agents_dir, sym, frm)
    paths["agents_root"] = paths["agents"]
    paths["results"] = _mk(results_dir, sym, frm, run_base)
    paths["logs"] = _mk(logs_root, sym, frm, run_base)
    paths["reports"] = _mk(reports_dir, sym, frm, run_base)

    paths["error_log"] = os.path.join(paths["logs"], "error.log")
    paths["benchmark_log"] = os.path.join(paths["logs"], "benchmark.log")
    paths["train_log"] = os.path.join(paths["logs"], "train_log.csv")
    paths["risk_log"] = os.path.join(paths["logs"], "risk_log.csv")
    paths["risk_csv"] = os.path.join(paths["logs"], "risk.csv")
    paths["signals_log"] = os.path.join(paths["logs"], "signals_log.csv")
    paths["callbacks_log"] = os.path.join(paths["logs"], "callbacks_log.csv")
    paths["decisions_jsonl"] = os.path.join(paths["logs"], "entry_decisions.jsonl")
    paths["tb_dir"] = os.path.join(paths["logs"], "events")

    paths["step_csv"] = os.path.join(paths["logs"], "step_log.csv")
    paths["steps_csv"] = paths["step_csv"]
    paths["reward_csv"] = os.path.join(paths["results"], "reward", "reward.log")
    os.makedirs(os.path.dirname(paths["reward_csv"]), exist_ok=True)
    paths["train_csv"] = os.path.join(paths["logs"], "train_log.csv")
    paths["eval_csv"] = os.path.join(paths["logs"], "evaluation.csv")
    paths["trade_csv"] = os.path.join(paths["logs"], "deep_rl_trades.csv")
    paths["trades_csv"] = paths["trade_csv"]

    paths["memory_file"] = DEFAULT_MEMORY_FILE
    paths["kb_file"] = DEFAULT_KB_FILE

    paths["model_zip"] = os.path.join(paths["agents"], "deep_rl.zip")
    paths["model_best_zip"] = os.path.join(paths["agents"], "deep_rl_best.zip")
    vecnorm_file = os.path.join(paths["agents"], "vecnorm.pkl")
    paths["vecnorm_pkl"] = vecnorm_file
    paths["vecnorm"] = vecnorm_file
    paths["vecnorm_best"] = vecnorm_file
    paths["vecnorm_last"] = vecnorm_file
    paths["best_meta"] = os.path.join(paths["agents"], "best_meta.json")
    paths["last_meta"] = os.path.join(paths["agents"], "last_meta.json")

    return paths


def setup_logging(paths: dict) -> None:
    import json
    import logging
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    def add_file_handler(path: str, level: int) -> None:
        fh = RotatingFileHandler(path, maxBytes=50_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    add_file_handler(paths["train_log"], logging.INFO)
    add_file_handler(paths["benchmark_log"], logging.INFO)

    errh = RotatingFileHandler(paths["error_log"], maxBytes=50_000_000, backupCount=5, encoding="utf-8")
    errh.setLevel(logging.ERROR)
    errh.setFormatter(fmt)
    root.addHandler(errh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    risk_logger = logging.getLogger("config.risk_manager")
    risk_logger.setLevel(logging.INFO)
    risk_fh = RotatingFileHandler(paths["risk_log"], maxBytes=50_000_000, backupCount=5, encoding="utf-8")
    risk_fh.setLevel(logging.INFO)
    risk_fh.setFormatter(fmt)
    risk_logger.addHandler(risk_fh)
    risk_logger.propagate = False


def ensure_state_files(memory_file: str, kb_file: str) -> None:
    mem_dir = os.path.dirname(memory_file) or ""
    kb_dir = os.path.dirname(kb_file) or ""
    if mem_dir:
        os.makedirs(mem_dir, exist_ok=True)
    if kb_dir:
        os.makedirs(kb_dir, exist_ok=True)

    if not os.path.exists(memory_file):
        mem_init = {"sessions": {}, "ai_trace": []}
        with ensure_utf8(memory_file, csv_newline=False) as fh:
            import json

            json.dump(mem_init, fh, ensure_ascii=False, indent=2)

    if not os.path.exists(kb_file):
        kb_init = {
            "version": "2.0",
            "strategy_memory": {},
            "skills": {
                "strong_frames": [],
                "weak_frames": [],
                "preferred_entry_signals": [],
                "danger_signals": [],
            },
            "learning_parameters": {
                "reward_weights": {},
                "risk": {},
            },
            "risk": {},
            "performance": {},
            "meta": {},
        }
        with ensure_utf8(kb_file, csv_newline=False) as fh:
            import json

            json.dump(kb_init, fh, ensure_ascii=False, indent=2)


def state_paths_from_env() -> dict:
    return {"memory_file": DEFAULT_MEMORY_FILE, "kb_file": DEFAULT_KB_FILE}


def get_paths(symbol: str, frame: str) -> dict:
    sym = symbol.upper()
    frm = str(frame)
    base = _mk(DEFAULT_RESULTS_DIR, sym, frm)
    logs_dir = _mk(base, "logs")
    agents_dir = _mk(DEFAULT_AGENTS_DIR, sym, frm)
    return {
        "base": base,
        "logs_dir": logs_dir,
        "train_csv": os.path.join(logs_dir, "train_log.csv"),
        "eval_csv": os.path.join(logs_dir, "evaluation.csv"),
        "reward_csv": os.path.join(logs_dir, "reward.csv"),
        "trade_csv": os.path.join(logs_dir, "deep_rl_trades.csv"),
        "trades_csv": os.path.join(logs_dir, "deep_rl_trades.csv"),
        "step_csv": os.path.join(logs_dir, "step_log.csv"),
        "jsonl_decisions": os.path.join(logs_dir, "entry_decisions.jsonl"),
        "benchmark_log": os.path.join(logs_dir, "benchmark.log"),
        "risk_log": os.path.join(logs_dir, "risk.log"),
        "risk_csv": os.path.join(logs_dir, "risk.csv"),
        "report_dir": _mk(DEFAULT_RESULTS_DIR, "reports"),
        "perf_dir": _mk(DEFAULT_RESULTS_DIR, "performance"),
        "best_zip": os.path.join(agents_dir, "deep_rl_best.zip"),
    }


__all__ = [
    "get_root",
    "memory_dir",
    "algo_scoped",
    "agents_dir",
    "latest_agent",
    "best_agent",
    "vecnorm_path",
    "results_dir",
    "reports_dir",
    "logs_dir",
    "dataset_path",
    "ensure_utf8",
    "_atomic_replace",
    "stamp_name",
    "RunPaths",
    "new_run_id",
    "ensure_contract",
    "build_paths",
    "setup_logging",
    "ensure_state_files",
    "state_paths_from_env",
    "get_paths",
]

