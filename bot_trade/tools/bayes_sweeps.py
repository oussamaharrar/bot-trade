from __future__ import annotations
"""Bayesian hyper-parameter sweeper using Optuna."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import optuna  # type: ignore
import yaml

from bot_trade.tools.atomic_io import write_json, write_png
from bot_trade.tools.force_utf8 import force_utf8


def _objective(trial: optuna.Trial, space: Dict[str, Any]) -> float:
    for name, spec in space.items():
        if spec["type"] == "float":
            trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "int":
            trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            trial.suggest_categorical(name, spec["choices"])
    sharpe = random.random()
    trial.set_user_attr("metrics", {"sharpe": sharpe})
    return sharpe


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(description="Bayesian sweeps")
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--study-name", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    study = optuna.create_study(direction="maximize", study_name=args.study_name, storage=f"sqlite:///results/sweeps/{args.study_name}.db")
    study.optimize(lambda t: _objective(t, cfg["space"]), n_trials=cfg.get("n_trials", 10))

    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]
    winners = []
    for t in trials:
        winners.append({"trial": t.number, **t.user_attrs.get("metrics", {})})
    out_dir = Path("results") / "sweeps" / args.study_name
    write_json(out_dir / "winners.json", winners)
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    write_png(out_dir / "charts" / "progress.png", fig.figure)


if __name__ == "__main__":
    main()
