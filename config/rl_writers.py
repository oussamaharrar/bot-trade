import os, csv, json, threading
from typing import Optional, Union, Iterable
from datetime import datetime

# ==============================================
# Writers (CSV / JSONL) — Windows-safe, thread-safe
# Fixes: initialize CSV writer before header write,
#        accept dict rows mapped to header, add newlines for JSONL,
#        robust flush/close.
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
        # 1) open file & csv writer first
        self._fh = open(self.path, mode="a", encoding="utf-8", newline="")
        self._csv = csv.writer(self._fh)
        # 2) write header once if file is empty
        self._maybe_write_header()

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
            if isinstance(row, dict):
                row = self._from_dict(row)
            self._csv.writerow(row)
            self._fh.flush()

    def flush(self):
        with self._lock:
            try:
                self._fh.flush()
            except Exception:
                pass

    def close(self):
        with self._lock:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass

class JSONLWriter(_BaseWriter):
    def __init__(self, path: str):
        super().__init__(path, header=None)
        self._fh = open(self.path, mode="a", encoding="utf-8", newline="\n")

    def write(self, obj: dict):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def flush(self):
        with self._lock:
            try:
                self._fh.flush()
            except Exception:
                pass

    def close(self):
        with self._lock:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass

class WritersBundle:
    """
    حزمة كتّاب موحّدة يستخدمها rl_callbacks:
      .reward     -> CSVWriter    ["ts","frame","symbol","step","avg_reward","ep_rew_mean"]
      .trades     -> CSVWriter    ["ts","frame","symbol","step","side","price","size","pnl","equity","reason"]
      .benchmark  -> CSVWriter    ["ts","frame","symbol","step","fps","cpu_percent","ram_gb","gpu0_mb","gpu1_mb","gpu2_mb","gpu3_mb"]
      .error      -> CSVWriter    ["ts","step","message"]
      .train      -> CSVWriter    ["ts","frame","symbol","file","timesteps","status"]
      .eval       -> CSVWriter    ["ts","frame","symbol","metric","value"]
      .tb         -> SummaryWriter (اختياري)
      .decisions  -> JSONLWriter  (اختياري) قرارات الدخول المنظمة
    """
    def __init__(self, paths: dict, enable_tb: bool = False, tb_dir: Optional[str] = None):
        self.paths = paths
        # Ensure base dirs exist
        for key in ("results", "logs"):
            if key in paths:
                os.makedirs(paths[key], exist_ok=True)

        # CSV writers (headers aligned with callbacks)
        self.reward    = CSVWriter(paths["reward_csv"], header=["ts","frame","symbol","step","avg_reward","ep_rew_mean"])
        self.trades    = CSVWriter(paths["trade_csv"],  header=["ts","frame","symbol","step","side","price","size","pnl","equity","reason"])
        self.benchmark = CSVWriter(paths["benchmark_log"], header=["ts","frame","symbol","step","fps","cpu_percent","ram_gb","gpu0_mb","gpu1_mb","gpu2_mb","gpu3_mb"])
        # store miscellaneous logs inside the dedicated logs directory
        self.error     = CSVWriter(os.path.join(self.paths.get("logs", self.paths.get("results")), "error.csv"), header=["ts","step","message"])
        self.train     = CSVWriter(self.paths["train_csv"], header=["ts","frame","symbol","file","timesteps","status"]) 
        self.eval      = CSVWriter(self.paths["eval_csv"],  header=["ts","frame","symbol","metric","value"]) 

        # JSONL decisions
        try:
            self.decisions = JSONLWriter(os.path.join(self.paths["logs"], "entry_decisions.jsonl"))
        except Exception:
            self.decisions = None

        # TensorBoard (optional)
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
        for w in (self.reward, self.trades, self.benchmark, self.error, self.train, self.eval):
            w.flush()
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

    def close(self):
        self.flush()
        for w in (self.reward, self.trades, self.benchmark, self.error, self.train, self.eval):
            w.close()
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


def build_writers(paths: dict, enable_tb: bool = False) -> WritersBundle:
    """أنشئ الحزمة الموحّدة للكتّاب بناءً على مسارات rl_paths.build_paths(...)"""
    return WritersBundle(paths, enable_tb=enable_tb, tb_dir=None)

# Alias لضمان التوافق مع أي استيراد قديم
class Writers(WritersBundle):
    pass
