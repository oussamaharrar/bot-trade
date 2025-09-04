
from __future__ import annotations
"""Command template whitelist using :mod:`commands_registry`."""

from typing import Dict, List, Sequence

from . import commands_registry
ALIASES = {"train_rl": "train"}



class ValidationError(ValueError):
    """Raised when parameters do not match the whitelist."""


def build_command(name: str, params: Dict[str, object], extra_flags: Sequence[str] | None = None) -> List[str]:
    name = ALIASES.get(name, name)
    try:
        return commands_registry.build_command(name, params, extra_flags)
    except KeyError as exc:  # unknown command
        raise ValidationError(str(exc)) from exc
    except ValueError as exc:  # invalid flag or value
        raise ValidationError(str(exc)) from exc


def validate(cmd: List[str]) -> None:
    """Basic validation ensuring *cmd* matches a whitelisted template.

    The command must start with a registered template prefix and not contain
    shell metacharacters.
    """

    if not cmd:
        raise ValidationError("empty command")
    if cmd[0] != "python":
        raise ValidationError("commands must start with python")
    for tok in cmd:
        if any(c in tok for c in "|;&><"):
            raise ValidationError("unsafe token")
    # Ensure command corresponds to a known registry entry
    name_matched = False
    for tmpl in commands_registry.REGISTRY.values():
        temp = list(tmpl.template)
        if len(cmd) >= len(temp) and all(
            (t.startswith('{') and not c.startswith('-')) or t == c
            for t, c in zip(temp, cmd)
        ):
            name_matched = True
            break
    if not name_matched:
        raise ValidationError("command not whitelisted")


__all__ = ["build_command", "ValidationError", "validate"]
