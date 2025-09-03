from __future__ import annotations

"""Generate user config from defaults and optional presets."""

import argparse
from pathlib import Path
from typing import Dict

import yaml  # type: ignore

from bot_trade.config.encoding import force_utf8
from bot_trade.config.schema import Config

PRESETS_DIR = Path(__file__).resolve().parents[1] / "config" / "presets"
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "config" / "config.default.yaml"


def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def apply_presets(cfg: Dict, training: str | None, net: str | None) -> Dict:
    if training:
        training_map = _load_yaml(PRESETS_DIR / "training.yml")
        cfg.update(training_map.get(training, {}))
    if net:
        net_map = _load_yaml(PRESETS_DIR / "net_arch.yml")
        cfg.setdefault("net", {}).update(net_map.get(net, {}))
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create config from defaults and presets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", action="append", default=[], help="training=name or net=name")
    ns = ap.parse_args(argv)

    training = net = None
    for item in ns.preset:
        if "=" in item:
            k, v = item.split("=", 1)
            if k == "training":
                training = v
            elif k == "net":
                net = v
    cfg = _load_yaml(DEFAULT_CFG)
    cfg = apply_presets(cfg, training, net)
    Config.model_validate(cfg)  # type: ignore
    out_path = Path(ns.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_dump(cfg, sort_keys=False)
    if not data.endswith("\n"):
        data += "\n"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(out_path)
    print(f"[CONFIG] wrote={out_path} preset.training={training} preset.net={net}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
