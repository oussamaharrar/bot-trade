from __future__ import annotations

"""Generate user config from defaults and optional presets."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml  # type: ignore

from bot_trade.tools.encoding import force_utf8
from bot_trade.config.schema import Config

PRESETS_DIR = Path(__file__).resolve().parents[1] / "config" / "presets"
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "config" / "config.default.yaml"


def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _apply_presets(cfg: Dict, presets: List[Tuple[str, str]]) -> Dict:
    for key, name in presets:
        mapping = _load_yaml(PRESETS_DIR / f"{key}.yml")
        if key == "net":
            cfg.setdefault("net", {}).update(mapping.get(name, {}))
        else:
            cfg.update(mapping.get(name, {}))
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create config from defaults and presets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", action="append", default=[], help="key=value overlay; multiple allowed")
    ap.add_argument("--force", action="store_true", help="Overwrite if destination exists")
    ns = ap.parse_args(argv)

    presets: List[Tuple[str, str]] = []
    for item in ns.preset:
        if "=" in item:
            k, v = item.split("=", 1)
            presets.append((k, v))

    cfg = _load_yaml(DEFAULT_CFG)
    cfg = _apply_presets(cfg, presets)
    Config.model_validate(cfg)  # type: ignore

    out_path = Path(ns.out)
    if out_path.exists() and not ns.force:
        print(f"[CONFIG] out={out_path} exists=true valid=false")
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_dump(cfg, sort_keys=False)
    if not data.endswith("\n"):
        data += "\n"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(out_path)
    print(
        f"[CONFIG] out={out_path} presets={[f'{k}={v}' for k, v in presets]} valid=true"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    force_utf8()
    raise SystemExit(main())
