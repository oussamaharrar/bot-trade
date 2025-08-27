import os, sys, json, logging
from logging.handlers import RotatingFileHandler

"""
rl_paths.py — توحيد المسارات والملفات العامة للنظام.
- يدعم Overrides عبر متغيرات البيئة (اختياري):
  BOT_AGENTS_DIR, BOT_RESULTS_DIR, BOT_REPORTS_DIR, BOT_MEMORY_FILE, BOT_KB_FILE
- يبني مسارات جلسة التدريب (حسب symbol/frame) ويضيف مخزونًا غنيًا للملفات.
- يوفّر دوال مساعدة لضمان وجود ملفات الحالة بهيكل افتراضي متوافق مع ai_core.
"""

DEFAULT_AGENTS_DIR  = os.environ.get("BOT_AGENTS_DIR",  "agents")
DEFAULT_RESULTS_DIR = os.environ.get("BOT_RESULTS_DIR", "results")
DEFAULT_REPORTS_DIR = os.environ.get("BOT_REPORTS_DIR", "reports")
DEFAULT_MEMORY_FILE = os.environ.get("BOT_MEMORY_FILE", os.path.join("memory", "memory.json"))
DEFAULT_KB_FILE     = os.environ.get("BOT_KB_FILE",     os.path.join("memory", "knowledge_base_full.json"))

def _mk(*parts):
    p = os.path.join(*parts)
    os.makedirs(p, exist_ok=True)
    return p

def build_paths(symbol: str, frame: str,
                agents_dir: str = None,
                results_dir: str = None,
                reports_dir: str = None):
        # Resolve dirs with environment overrides if None passed
    agents_dir  = agents_dir  or DEFAULT_AGENTS_DIR
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    reports_dir = reports_dir or DEFAULT_REPORTS_DIR

    sym, frm = symbol.upper(), str(frame)
    paths = {}
    paths["agents"]  = _mk(agents_dir,  sym, frm)
    paths["results"] = _mk(results_dir, sym, frm)
    paths["reports"] = _mk(reports_dir, sym, frm)
    paths["logs"]    = _mk(paths["results"], "logs")

        # logs
    paths["error_log"]       = os.path.join(paths["logs"], "error.log")
    paths["benchmark_log"]   = os.path.join(paths["logs"], "benchmark.log")
    paths["train_log"]       = os.path.join(paths["logs"], f"train_rl_{frm}.log")
    paths["risk_log"]        = os.path.join(paths["logs"], "risk_manager.log")  # للـ logging القياسي
    paths["risk_csv"]        = os.path.join(paths["logs"], "risk.csv")          # لكتابة CSV من RiskManager
    paths["decisions_jsonl"] = os.path.join(paths["logs"], "entry_decisions.jsonl")
    paths["tb_dir"]          = os.path.join(paths["results"], "tb")

        # csv
    paths["steps_csv"]   = os.path.join(paths["results"], f"steps_{frm}.csv")
    paths["reward_csv"]  = os.path.join(paths["results"], f"reward_{frm}.csv")
    paths["train_csv"]   = os.path.join(paths["results"], "train_log.csv")
    paths["eval_csv"]    = os.path.join(paths["results"], "evaluation.csv")
    paths["trade_csv"]   = os.path.join(paths["results"], "deep_rl_trades.csv")

    # state files (global)
    paths["memory_file"] = DEFAULT_MEMORY_FILE
    paths["kb_file"]     = DEFAULT_KB_FILE

    # models / vecnorm
    paths["model_zip"]      = os.path.join(paths["agents"], "deep_rl.zip")
    paths["model_best_zip"] = os.path.join(paths["agents"], "deep_rl_best.zip")
    paths["vecnorm_pkl"]    = os.path.join(paths["agents"], "vecnorm.pkl")
    paths["vecnorm_best"]   = os.path.join(paths["agents"], "vecnorm_best.pkl")
    paths["best_meta"]      = os.path.join(paths["agents"], "best_ckpt.json")
    
    # state files
    paths["memory_file"] = DEFAULT_MEMORY_FILE
    paths["kb_file"]     = DEFAULT_KB_FILE
    return paths


def get_paths(symbol: str, frame: str) -> dict:
    """Return a simplified dictionary of important file paths.

    The returned dict matches the keys used by :class:`UpdateManager` and
    other high level utilities. All paths are relative to the repository
    root and created on demand.
    """

    paths = build_paths(symbol, frame)
    # ensure directories required by UpdateManager
    os.makedirs(paths["logs"], exist_ok=True)
    os.makedirs(paths["results"], exist_ok=True)
    os.makedirs(paths["reports"], exist_ok=True)

    out = {
        "base": paths["results"],
        "train_csv": paths["train_csv"],
        "eval_csv": paths["eval_csv"],
        "trades_csv": paths["trade_csv"],
        "step_csv": paths["steps_csv"],
        "logs_dir": paths["logs"],
        "jsonl_decisions": paths["decisions_jsonl"],
        "benchmark_log": paths["benchmark_log"],
        "risk_log": paths["risk_log"],
        "report_dir": paths["reports"],
        "perf_dir": os.path.join(paths["results"], "performance"),
        "best_zip": paths["model_best_zip"],
    }
    # create performance directory lazily
    os.makedirs(out["perf_dir"], exist_ok=True)
    return out

def setup_logging(paths: dict):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    def add_file_handler(path, level):
        fh = RotatingFileHandler(path, maxBytes=50_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(level); fh.setFormatter(fmt); root.addHandler(fh)

    add_file_handler(paths["train_log"],     logging.INFO)
    add_file_handler(paths["benchmark_log"], logging.INFO)

    errh = RotatingFileHandler(paths["error_log"], maxBytes=50_000_000, backupCount=5, encoding="utf-8")
    errh.setLevel(logging.ERROR); errh.setFormatter(fmt); root.addHandler(errh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt); root.addHandler(sh)

    # risk logger
    risk_logger = logging.getLogger("config.risk_manager")
    risk_logger.setLevel(logging.INFO)
    risk_fh = RotatingFileHandler(paths["risk_log"], maxBytes=50_000_000, backupCount=5, encoding="utf-8")
    risk_fh.setLevel(logging.INFO); risk_fh.setFormatter(fmt); risk_logger.addHandler(risk_fh)
    risk_logger.propagate = False

def ensure_state_files(memory_file: str, kb_file: str):
    """توليد ملفات الحالة بهياكل افتراضية متوافقة مع ai_core/self_improver.
    - memory.json: يحتوي sessions + ai_trace
    - knowledge_base_full.json: يحتوي strategy_memory + skills + learning_parameters + risk … إلخ
    """
    mem_dir = os.path.dirname(memory_file) or ""
    kb_dir  = os.path.dirname(kb_file) or ""
    if mem_dir:
        os.makedirs(mem_dir, exist_ok=True)
    if kb_dir:
        os.makedirs(kb_dir, exist_ok=True)

    if not os.path.exists(memory_file):
        mem_init = {"sessions": {}, "ai_trace": []}
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(mem_init, f, ensure_ascii=False, indent=2)

    if not os.path.exists(kb_file):
        kb_init = {
            "version": "2.0",
            "strategy_memory": {},
            "skills": {
                "strong_frames": [],
                "weak_frames": [],
                "preferred_entry_signals": [],
                "danger_signals": []
            },
            "learning_parameters": {
                "reward_weights": {},
                "risk": {}
            },
            "risk": {},
            "performance": {},
            "meta": {}
        }
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(kb_init, f, ensure_ascii=False, indent=2)


def state_paths_from_env() -> dict:
    """أرجِع مسارات الحالة مع تطبيق Overrides من البيئة.
    مفيد لتمريرها إلى Train_RL/Callbacks دون توزيع معرفة المسارات في كل ملف.
    """
    return {"memory_file": DEFAULT_MEMORY_FILE, "kb_file": DEFAULT_KB_FILE}
