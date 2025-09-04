from __future__ import annotations
"""Safe command templates for the panel.

Exposes ``REGISTRY`` and ``build_command`` for constructing commands from
whitelisted templates with strict placeholder substitution.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence
import shlex


@dataclass
class CommandTemplate:
    template: Sequence[str]
    allowed_flags: Sequence[str] | None = None
    defaults: Mapping[str, str] | None = None


REGISTRY: Dict[str, CommandTemplate] = {
    "train": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.train_rl",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--total-steps",
            "{total_steps}",
            "--n-envs",
            "{n_envs}",
            "--device",
            "{device}",
            "--headless",
            "--data-dir",
            "{data_dir}",
            "{allow_synth}",
            "{resume_auto}",
            "{vecnorm}",
        ],
        allowed_flags=[
            "--policy",
            "--net-arch",
            "--epochs",
            "--batch-size",
            "--sb3-verbose",
            "--log-level",
            "--checkpoint-every",
            "--eval-every-steps",
            "--eval-episodes",
        ],
        defaults={
            "allow_synth": "--allow-synth",
            "resume_auto": "--resume-auto",
            "vecnorm": "--vecnorm",
        },
    ),
    "eval_run": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.tools.eval_run",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--run-id",
            "{run_id}",
            "--tearsheet",
        ]
    ),
    "export_charts": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.tools.monitor_manager",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--run-id",
            "{run_id}",
            "--no-wait",
            "--headless",
        ]
    ),
    "wfa_gate": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.eval.wfa_gate",
            "--config",
            "{wfa_cfg}",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--windows",
            "{windows}",
            "--embargo",
            "{embargo}",
            "--profile",
            "{profile}",
        ]
    ),
    "bayes_sweeps": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.tools.sweep",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--grid",
            "{grid_or_yaml}",
        ]
    ),
    "live_dry_run": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.runners.live_dry_run",
            "--exchange",
            "{exchange}",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--gateway",
            "paper",
            "--duration",
            "{duration}",
            "--model-optional",
            "--bootstrap-price",
            "{bootstrap_price}",
        ]
    ),
    "paths_doctor": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.tools.paths_doctor",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--strict",
        ]
    ),
    "gen_synth_data": CommandTemplate(
        template=[
            "python",
            "-m",
            "bot_trade.tools.gen_synth_data",
            "--symbol",
            "{symbol}",
            "--frame",
            "{frame}",
            "--days",
            "{days}",
            "--out",
            "{out_dir}",
        ]
    ),
}


def _validate_value(name: str, value: object) -> str:
    s = str(value)
    if any(c in s for c in "|;&><"):
        raise ValueError(f"unsafe character in {name}")
    if " " in s and not s.isdigit():
        raise ValueError(f"spaces not allowed in {name}")
    return s


def build_command(name: str, params: Mapping[str, object], extra_flags: Sequence[str] | None = None) -> List[str]:
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
        for flg in extra_flags:
            parts = shlex.split(flg)
            for part in parts:
                if part and part.split('=')[0] not in allowed:
                    raise ValueError(f"flag {part} not allowed")
            flags.extend(parts)
    return [tok for tok in result if tok] + flags


__all__ = ["REGISTRY", "build_command", "CommandTemplate"]
