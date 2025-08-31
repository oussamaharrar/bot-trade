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

    def _from_dict(self, row_dict: dict) -> list:
        if self._header:
            return [row_dict.get(k, "") for k in self._header]
        return list(row_dict.values())

    def write(self, row: Union[dict, Iterable]):
        if isinstance(row, dict):
            row = self._from_dict(row)
        with self._lock:
            existing = ""
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    existing = fh.read().rstrip("\n")
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="") as fh:
                if existing:
                    fh.write(existing + "\n")
                elif self._header:
                    csv.writer(fh).writerow(self._header)
                csv.writer(fh).writerow(row)
            os.replace(tmp, self.path)

    def flush(self):
        pass

    def close(self):
        pass


class RunIDCSVWriter(CSVWriter):
    """CSVWriter that prepends a fixed ``run_id`` column."""

    def __init__(self, path: str, run_id: str, header: Optional[list] = None):
        header = ["run_id"] + (header or [])
        super().__init__(path, header=header)
        self._run_id = run_id

    def write(self, row: Union[dict, Iterable]):  # type: ignore[override]
        if isinstance(row, dict):
            row = {"run_id": self._run_id, **row}
        else:
            row = [self._run_id, *row]
        super().write(row)

class JSONLWriter(_BaseWriter):
    def __init__(self, path: str):
        super().__init__(path, header=None)

    def write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            existing = ""
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    existing = fh.read().rstrip("\n")
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                if existing:
                    fh.write(existing + "\n")
                fh.write(line + "\n")
            os.replace(tmp, self.path)

    def flush(self):
        pass

    def close(self):
        pass


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
        reward_val = row.get("reward_total", row.get("reward"))
        if reward_val in (None, ""):
            reward_val = 0.0
        base = {
            "run_id": self.run_id,
            "ts": row.get("ts", ""),
            "global_step": row.get("global_step", ""),
            "env_idx": row.get("env_idx", ""),
            "reward_total": reward_val,
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

        self.trades = RunIDCSVWriter(
            paths["trade_csv"],
            run_id,
            header=["ts", "frame", "symbol", "step", "side", "price", "size", "pnl", "equity", "reason"],
        )
        self.benchmark = RunIDCSVWriter(
            paths["benchmark_log"],
            run_id,
            header=["ts", "frame", "symbol", "step", "fps", "cpu_percent", "ram_gb", "gpu0_mb", "gpu1_mb", "gpu2_mb", "gpu3_mb"],
        )
        self.error = RunIDCSVWriter(os.path.join(logs_dir, "error.csv"), run_id, header=["ts", "step", "message"])
        self.train = RunIDCSVWriter(
            paths["train_csv"],
            run_id,
            header=["ts", "frame", "symbol", "file", "timesteps", "status"],
        )
        self.eval = RunIDCSVWriter(paths["eval_csv"], run_id, header=["ts", "frame", "symbol", "metric", "value"])
        self.signals = RunIDCSVWriter(paths["signals_log"], run_id, header=["ts", "event", "detail"])
        self.callbacks = RunIDCSVWriter(paths["callbacks_log"], run_id, header=["ts", "callback", "action"])

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
                base = {"run_id": self.run_id, "ts": now_iso()}
                base.update(payload)
                self.decisions.write(base)
        except Exception:
            pass

    def flush(self):
        for w in (self.trades, self.benchmark, self.error, self.train, self.eval, self.signals, self.callbacks):
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
        for w in (self.trades, self.benchmark, self.error, self.train, self.eval, self.signals, self.callbacks):
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
