from __future__ import annotations
"""Command registry loaded from YAML templates."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence
import shlex
import yaml


@dataclass
class CommandTemplate:
    template: Sequence[str]
    allowed_flags: Sequence[str] | None = None
    defaults: Mapping[str, str] | None = None
    forbidden_flags: Sequence[str] | None = None


def _load_registry() -> Dict[str, CommandTemplate]:
    path = Path(__file__).with_name("commands_registry.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    reg: Dict[str, CommandTemplate] = {}
    for name, entry in data.items():
        reg[name] = CommandTemplate(
            template=entry.get("template", []),
            allowed_flags=entry.get("allowed_flags"),
            defaults=entry.get("defaults"),
            forbidden_flags=entry.get("forbidden_flags"),
        )
    return reg


REGISTRY: Dict[str, CommandTemplate] = _load_registry()


def _validate_value(name: str, value: object) -> str:
    s = str(value)
    if any(c in s for c in "|;&><"):
        raise ValueError(f"unsafe character in {name}")
    if " " in s and not s.isdigit():
        raise ValueError(f"spaces not allowed in {name}")
    return s


def build_command(
    name: str,
    params: Mapping[str, object],
    extra_flags: Sequence[str] | None = None,
) -> List[str]:
    if name not in REGISTRY:
        raise KeyError(name)
    ct = REGISTRY[name]
    values: Dict[str, str] = {}
    placeholders = {tok[1:-1] for tok in ct.template if tok.startswith('{') and tok.endswith('}')}
    if ct.defaults:
        values.update(ct.defaults)
        placeholders.update(ct.defaults.keys())
    unknown = set(params) - placeholders
    if unknown:
        raise ValueError(f"unknown params: {sorted(unknown)}")
    for k, v in params.items():
        values[k] = _validate_value(k, v)
    result: List[str] = []
    for tok in ct.template:
        if tok.startswith('{') and tok.endswith('}'):
            key = tok[1:-1]
            val = values.get(key, '')
            if val:
                result.append(val)
        else:
            result.append(tok)
    flags: List[str] = []
    if extra_flags:
        allowed = set(ct.allowed_flags or [])
        forbidden = set(ct.forbidden_flags or [])
        for flg in extra_flags:
            parts = shlex.split(flg)
            for part in parts:
                base = part.split('=')[0]
                if base in forbidden:
                    raise ValueError(f"flag {base} forbidden")
                if allowed and base not in allowed:
                    raise ValueError(f"flag {base} not allowed")
            flags.extend(parts)
    return [tok for tok in result if tok] + flags


__all__ = ["REGISTRY", "build_command", "CommandTemplate"]
