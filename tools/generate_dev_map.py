"""Generate developer-oriented architecture map in Markdown."""
from __future__ import annotations

from tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import argparse
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCK = """
[CLI] -> [Config] -> [Data Loading] -> [Env] -> [Agent] -> [Writers] -> [Memory] -> [Reports/Results] -> [Monitors]
"""


def module_inventory() -> list[str]:
    mods = []
    for pkg in ["ai_core", "config", "tools"]:
        p = ROOT / pkg
        for py in p.rglob("*.py"):
            rel = py.relative_to(ROOT).with_suffix("")
            mods.append(str(rel).replace("/", "."))
    return sorted(mods)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dev map")
    args = ap.parse_args()
    mods = module_inventory()

    print("# Bot Architecture Map\n")
    print("## High-level Flow\n")
    print("```")
    print(BLOCK.strip())
    print("```")
    print("\n## Module Inventory\n")
    for m in mods:
        print(f"- {m}")
    print("\n## Runtime Flow\n")
    flow = [
        "Parse CLI args",
        "Load configuration",
        "Build data loaders and env",
        "Instantiate agent",
        "Train/evaluate with writers",
        "Persist memory and knowledge",
        "Generate reports and launch monitors",
    ]
    for i, step in enumerate(flow, 1):
        print(f"{i}. {step}")
    print("\n## Inputs/Outputs\n")
    print("- configs: config/*.py")
    print("- data: results/{symbol}/{frame}/")
    print("- reports: report/{symbol}/{frame}/{run_id}/")
    print("- logs: logs/{symbol}/{frame}/{run_id}/")
    print("\n## Environment Variables\n")
    print("- MONITOR_SHELL, MONITOR_DEBUG, MONITOR_USE_CONDA_RUN")
    print("\n## Adding new tools\n")
    print("Place new scripts under tools/ and ensure absolute imports.")


if __name__ == "__main__":
    main()
