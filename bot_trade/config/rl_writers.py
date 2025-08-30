import os, csv, json, threading
from typing import Optional, Union, Iterable, Dict, Any
from datetime import datetime
from pathlib import Path

# ==============================================
# Writers (CSV / JSONL) — Windows-safe, thread-safe
# Fixes: initialize CSV writer before header write,
#        accept dict rows mapped to header, add newlines for JSONL,
#        robust flush/close with lazy reopen and idempotent close().
# ==============================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

class _BaseWriter:
    def __init__(self, path: str, header: Optional[list] = None):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._header = header or []
        # NOTE: subclasses are responsible for opening files & writing header
        self._fh = None  # type: ignore

    # Subclasses should implement: write(), flush(), close()

class CSVWriter(_BaseWriter):
    def __init__(self, path: str, header: Optional[list] = None):
        super().__init__(path, header=header)
        self._fh = None
        self._csv = None
        self._open()

    def _open(self) -> None:
        """(Re)open the underlying file handle as needed."""
        if self._fh is None or self._fh.closed:
            self._sanitize_tail()
            self._fh = open(self.path, mode="a", encoding="utf-8", newline="")
            self._csv = csv.writer(self._fh)
            self._maybe_write_header()

    def _sanitize_tail(self) -> None:
        """Remove blank/partial trailing lines before appending."""
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "rb+") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size == 0:
                    return
                pos = size - 1
                fh.seek(pos)
                last = fh.read(1)
                # strip trailing newlines
                while pos >= 0 and last in b"\r\n":
                    pos -= 1
                    fh.seek(pos)
                    last = fh.read(1)
                fh.truncate(pos + 1)
                fh.seek(pos + 1)
                fh.write(b"\n")
        except Exception:
            pass

    def _maybe_write_header(self) -> None:
        if not self._header:
            return
        try:
            need = (not os.path.exists(self.path)) or (os.path.getsize(self.path) == 0)
        except Exception:
            need = True
        if need:
            with self._lock:
                self._csv.writerow(self._header)
                self._fh.flush()

    def _from_dict(self, row_dict: dict) -> list:
        if self._header:
            return [row_dict.get(k, "") for k in self._header]
        # fallback to values order (unstable but ok if no header)
        return list(row_dict.values())

    def write(self, row: Union[dict, Iterable]):
        with self._lock:
            self._open()
            if isinstance(row, dict):
                row = self._from_dict(row)
            self._csv.writerow(row)
            self._fh.flush()

    def flush(self):
        with self._lock:
            try:
                if self._fh is not None and not self._fh.closed:
                    self._fh.flush()
            except Exception:
                pass

    def close(self):
        with self._lock:
            try:
                if self._fh is not None and not self._fh.closed:
                    self._fh.flush()
            except Exception:
                pass
            try:
                if self._fh is not None and not self._fh.closed:
                    self._fh.close()
            except Exception:
                pass
            self._fh = None
            self._csv = None

class JSONLWriter(_BaseWriter):
    def __init__(self, path: str):
        super().__init__(path, header=None)
        self._fh = None
        self._open()

    def _open(self) -> None:
        if self._fh is None or self._fh.closed:
            self._sanitize_tail()
            self._fh = open(self.path, mode="a", encoding="utf-8", newline="\n")

    def _sanitize_tail(self) -> None:
        """Remove partial trailing line before appending."""
        try:
            if not os.path.exists(self.path):
                return
            with open(self.path, "rb+") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size == 0:
                    return
                pos = size - 1
                fh.seek(pos)
                last = fh.read(1)
                if last in b"\r\n":
                    while pos >= 0 and last in b"\r\n":
                        pos -= 1
                        fh.seek(pos)
                        last = fh.read(1)
                    fh.truncate(pos + 1)
                    fh.seek(pos + 1)
                    fh.write(b"\n")
                else:
                    while pos >= 0 and last not in b"\n\r":
                        pos -= 1
                        if pos < 0:
                            break
                        fh.seek(pos)
                        last = fh.read(1)
                    fh.truncate(pos + 1 if pos >= 0 else 0)
                    if pos >= 0:
                        fh.seek(pos + 1)
                        fh.write(b"\n")
        except Exception:
            pass

    def write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._open()
            self._fh.write(line + "\n")
            self._fh.flush()

    def flush(self):
        with self._lock:
            try:
                if self._fh is not None and not self._fh.closed:
                    self._fh.flush()
            except Exception:
                pass

    def close(self):
        with self._lock:
            try:
                if self._fh is not None and not self._fh.closed:
                    self._fh.flush()
            except Exception:
                pass
            try:
                if self._fh is not None and not self._fh.closed:
                    self._fh.close()
            except Exception:
                pass
            self._fh = None


class RewardWriter:
    """CSV writer specialised for per-step rewards.

    The first column is always ``run_id`` injected automatically from the
    owning :class:`WritersBundle`.

    # TODO: add CSV rotation after size X.
    # TODO: weekly aggregates for reward.log.
    """

    def __init__(self, path: Path, run_id: str):
        self.run_id = run_id
        self._writer = CSVWriter(
            str(path),
            header=[
                "run_id",
                "ts",
                "global_step",
                "env_idx",
                "reward_total",
                "pnl",
                "cost",
                "stability",
                "danger_pen",
            ],
        )

    def write_row(self, row: Dict[str, Any]) -> None:
        """Write a reward record tolerating missing components."""
        base = {
            "run_id": self.run_id,
            "ts": row.get("ts", ""),
            "global_step": row.get("global_step", ""),
            "env_idx": row.get("env_idx", ""),
            "reward_total": row.get("reward_total", row.get("reward", "")),
            "pnl": row.get("pnl", ""),
            "cost": row.get("cost", ""),
            "stability": row.get("stability", ""),
            "danger_pen": row.get("danger_pen", ""),
        }
        self._writer.write(base)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

class WritersBundle:
    """Unified bundle of writers used by training callbacks."""

    def __init__(self, paths: dict, run_id: str, enable_tb: bool = False, tb_dir: Optional[str] = None):
        self.paths = paths
        self.run_id = run_id

        for key in ("results", "logs"):
            if key in paths:
                os.makedirs(paths[key], exist_ok=True)

        logs_dir = self.paths.get("logs", self.paths.get("results"))

        self.trades = CSVWriter(
            paths["trade_csv"],
            header=["ts", "frame", "symbol", "step", "side", "price", "size", "pnl", "equity", "reason"],
        )
        self.benchmark = CSVWriter(
            paths["benchmark_log"],
            header=["ts", "frame", "symbol", "step", "fps", "cpu_percent", "ram_gb", "gpu0_mb", "gpu1_mb", "gpu2_mb", "gpu3_mb"],
        )
        self.error = CSVWriter(os.path.join(logs_dir, "error.csv"), header=["ts", "step", "message"])
        self.train = CSVWriter(
            paths["train_csv"],
            header=["run_id", "ts", "frame", "symbol", "file", "timesteps", "status"],
        )
        self.eval = CSVWriter(paths["eval_csv"], header=["ts", "frame", "symbol", "metric", "value"])

        try:
            self.reward = RewardWriter(Path(logs_dir) / "reward.log", run_id)
        except Exception:
            self.reward = None  # type: ignore

        try:
            self.decisions = JSONLWriter(os.path.join(logs_dir, "entry_decisions.jsonl"))
        except Exception:
            self.decisions = None

        self.tb = None
        if enable_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                tb_dir = tb_dir or os.path.join(paths["results"], "tb")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb = SummaryWriter(log_dir=tb_dir)
            except Exception:
                self.tb = None

    def log_decision(self, **payload):
        """سجّل قرارًا منسّقًا إلى JSONL داخل مجلد logs إن توفر."""
        try:
            if getattr(self, "decisions", None) is not None:
                base = {"ts": now_iso()}
                base.update(payload)
                self.decisions.write(base)
        except Exception:
            pass

    def flush(self):
        for w in (self.trades, self.benchmark, self.error, self.train, self.eval):
            w.flush()
        try:
            if getattr(self, "reward", None) is not None:
                self.reward.flush()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if getattr(self, "decisions", None) is not None:
                self.decisions.flush()
        except Exception:
            pass
        if self.tb is not None:
            try:
                self.tb.flush()
            except Exception:
                pass

    def flush_all(self):  # pragma: no cover - backwards alias
        self.flush()

    def close(self):
        self.flush()
        for w in (self.trades, self.benchmark, self.error, self.train, self.eval):
            w.close()
        try:
            if getattr(self, "reward", None) is not None:
                self.reward.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if getattr(self, "decisions", None) is not None:
                self.decisions.close()
        except Exception:
            pass
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass


def build_writers(paths: dict, run_id: str, enable_tb: bool = False) -> WritersBundle:
    """Construct :class:`WritersBundle` for the given ``run_id``."""
    return WritersBundle(paths, run_id, enable_tb=enable_tb, tb_dir=None)

# Alias لضمان التوافق مع أي استيراد قديم
class Writers(WritersBundle):
    pass
