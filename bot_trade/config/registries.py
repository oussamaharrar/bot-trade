from __future__ import annotations

"""Simple registries for configurable components."""

from typing import Dict, Any

_execution_configs: Dict[str, Dict[str, Any]] = {}


def register_execution(name: str, config: Dict[str, Any]) -> None:
    _execution_configs[name] = dict(config)


def get_execution(name: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return _execution_configs.get(name, {} if default is None else default)

