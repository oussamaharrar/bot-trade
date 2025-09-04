from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, List

WHITELIST_PATH = Path(__file__).with_name("whitelist.yaml")


def load_whitelist(path: Path = WHITELIST_PATH) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {str(k): list(v) for k, v in data.items()}


def build_command(name: str, params: Dict[str, str], whitelist: Dict[str, List[str]] | None = None) -> List[str]:
    wl = whitelist or load_whitelist()
    if name not in wl:
        raise ValueError(f"command '{name}' not in whitelist")
    template = wl[name]
    try:
        return [part.format(**params) for part in template]
    except KeyError as exc:  # pragma: no cover - user error
        missing = exc.args[0]
        raise ValueError(f"missing placeholder '{missing}' for command '{name}'") from exc
