"""Utility to generate config files with presets."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml  # type: ignore[import-untyped]

from bot_trade.tools.atomic_io import write_text
from bot_trade.config.schema import Config


PRESETS = {
    "default": Config().dict(),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate config file presets")
    ap.add_argument("--preset", default="default", choices=PRESETS.keys())
    ap.add_argument("--out", default="config.yaml")
    ns = ap.parse_args(argv)
    data = PRESETS[ns.preset]
    yaml_str = yaml.safe_dump(data, sort_keys=False)
    write_text(Path(ns.out), yaml_str)
    print(f"wrote {ns.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
