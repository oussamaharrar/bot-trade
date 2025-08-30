import os
import json
import pandas as pd
from typing import Dict

from bot_trade.config.rl_paths import get_paths


def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_decisions(symbol: str, frame: str) -> pd.DataFrame:
    paths = get_paths(symbol, frame)
    p = paths.get("jsonl_decisions")
    try:
        if p and os.path.exists(p):
            return pd.read_json(p, lines=True)
    except Exception:
        pass
    return pd.DataFrame()


def load_training_metrics(symbol: str, frame: str) -> Dict[str, pd.DataFrame]:
    paths = get_paths(symbol, frame)
    return {
        "train_log": _read_csv(paths.get("train_csv", "")),
        "evaluation": _read_csv(paths.get("eval_csv", "")),
        "step_log": _read_csv(paths.get("step_csv", "")),
        "reward": _read_csv(paths.get("reward_csv", "")),
    }


def load_trades(symbol: str, frame: str) -> pd.DataFrame:
    paths = get_paths(symbol, frame)
    return _read_csv(paths.get("trades_csv", ""))


def load_benchmark(symbol: str, frame: str) -> pd.DataFrame:
    paths = get_paths(symbol, frame)
    return _read_csv(paths.get("benchmark_log", ""))


def load_knowledge(path: str = "memory/knowledge_base_full.json") -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
