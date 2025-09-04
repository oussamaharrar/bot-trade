from __future__ import annotations

"""Command template whitelist for the control panel."""

import re
from typing import Callable, Dict, List

SYMBOL_RE = re.compile(r"^[A-Z0-9]+(USDT|USD|USDC)$")
FRAME_SET = {"1s","1m","3m","5m","15m","30m","1h","2h","4h","8h","12h","1d"}
DEVICE_SET = {"cpu","cuda:0","cuda:1","auto"}
RUN_ID_RE = re.compile(r"^(latest|[0-9a-fA-F-]{4,36})$")


class ValidationError(ValueError):
    """Raised when parameters do not match the whitelist."""


def _pos_int(val: object, name: str) -> int:
    try:
        iv = int(val)
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"{name} must be int") from exc
    if iv <= 0:
        raise ValidationError(f"{name} must be >0")
    return iv


def _bool_flag(params: Dict[str, object], name: str) -> List[str]:
    return [f"--{name}"] if params.get(name) else []

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def _train_rl(params: Dict[str, object]) -> List[str]:
    allowed = {
        "symbol","frame","total_steps","n_envs","device","headless","resume_auto",
        "allow_synth","data_dir","algorithm"
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValidationError(f"unknown flags: {sorted(unknown)}")
    symbol = params["symbol"]
    frame = params["frame"]
    if not SYMBOL_RE.match(str(symbol)):
        raise ValidationError("invalid symbol")
    if frame not in FRAME_SET:
        raise ValidationError("invalid frame")
    device = params.get("device","cpu")
    if device not in DEVICE_SET:
        raise ValidationError("invalid device")
    _pos_int(params.get("total_steps",1),"total_steps")
    _pos_int(params.get("n_envs",1),"n_envs")
    cmd = [
        "python","-m","bot_trade.train_rl",
        "--symbol",str(symbol),"--frame",str(frame),
        "--total-steps",str(params.get("total_steps",1)),
        "--n-envs",str(params.get("n_envs",1)),
        "--device",device,
    ]
    cmd += _bool_flag(params,"headless")
    cmd += _bool_flag(params,"resume_auto")
    cmd += _bool_flag(params,"allow_synth")
    if "data_dir" in params:
        cmd += ["--data-dir",str(params["data_dir"])]
    if "algorithm" in params:
        cmd += ["--algorithm",str(params["algorithm"])]
    return cmd


def _eval_run(params: Dict[str, object]) -> List[str]:
    allowed = {"symbol","frame","run_id","tearsheet"}
    unknown = set(params) - allowed
    if unknown:
        raise ValidationError(f"unknown flags: {sorted(unknown)}")
    symbol = params["symbol"]
    frame = params["frame"]
    run_id = params["run_id"]
    if not SYMBOL_RE.match(str(symbol)) or frame not in FRAME_SET:
        raise ValidationError("invalid symbol/frame")
    if not RUN_ID_RE.match(str(run_id)):
        raise ValidationError("invalid run_id")
    cmd = [
        "python","-m","bot_trade.tools.eval_run",
        "--symbol",symbol,"--frame",frame,"--run-id",run_id,
    ]
    cmd += _bool_flag(params,"tearsheet")
    return cmd


def _monitor_manager(params: Dict[str, object]) -> List[str]:
    allowed = {"symbol","frame","run_id","headless"}
    unknown = set(params) - allowed
    if unknown:
        raise ValidationError(f"unknown flags: {sorted(unknown)}")
    symbol = params["symbol"]
    frame = params["frame"]
    run_id = params["run_id"]
    if not SYMBOL_RE.match(str(symbol)) or frame not in FRAME_SET:
        raise ValidationError("invalid symbol/frame")
    if not RUN_ID_RE.match(str(run_id)):
        raise ValidationError("invalid run_id")
    cmd = [
        "python","-m","bot_trade.tools.monitor_manager",
        "--symbol",symbol,"--frame",frame,"--run-id",run_id,
    ]
    cmd += _bool_flag(params,"headless")
    return cmd


def _gen_synth_data(params: Dict[str, object]) -> List[str]:
    allowed = {"symbol","frame","days","out"}
    unknown = set(params) - allowed
    if unknown:
        raise ValidationError(f"unknown flags: {sorted(unknown)}")
    symbol = params["symbol"]
    frame = params["frame"]
    days = params["days"]
    out = params["out"]
    if not SYMBOL_RE.match(str(symbol)) or frame not in FRAME_SET:
        raise ValidationError("invalid symbol/frame")
    _pos_int(days,"days")
    cmd = [
        "python","-m","bot_trade.tools.gen_synth_data",
        "--symbol",symbol,"--frame",frame,"--days",str(days),"--out",str(out),
    ]
    return cmd


def _paths_doctor(params: Dict[str, object]) -> List[str]:
    allowed = {"symbol","frame","strict"}
    unknown = set(params) - allowed
    if unknown:
        raise ValidationError(f"unknown flags: {sorted(unknown)}")
    symbol = params["symbol"]
    frame = params["frame"]
    if not SYMBOL_RE.match(str(symbol)) or frame not in FRAME_SET:
        raise ValidationError("invalid symbol/frame")
    cmd = ["python","-m","bot_trade.tools.paths_doctor","--symbol",symbol,"--frame",frame]
    if params.get("strict"):
        cmd.append("--strict")
    return cmd


TEMPLATES: Dict[str, Callable[[Dict[str, object]], List[str]]] = {
    "train_rl": _train_rl,
    "eval_run": _eval_run,
    "monitor_manager": _monitor_manager,
    "gen_synth_data": _gen_synth_data,
    "paths_doctor": _paths_doctor,
}


def build_command(name: str, params: Dict[str, object]) -> List[str]:
    if name not in TEMPLATES:
        raise ValidationError("unknown command")
    return TEMPLATES[name](params)


__all__ = ["build_command", "ValidationError", "TEMPLATES"]
