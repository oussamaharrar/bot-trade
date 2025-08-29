"""Small CLI to inspect and manage training memory state."""
from __future__ import annotations

from tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import argparse
import json

from tools.memory_manager import MemoryManager


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect memory state")
    ap.add_argument("--print-latest", action="store_true", help="Print latest snapshot")
    ap.add_argument("--resume", action="store_true", help="Resume and print state")
    ap.add_argument("--compact", action="store_true", help="Write compact index entry")
    args = ap.parse_args()

    mm = MemoryManager()
    if args.print_latest or args.resume:
        state = mm.resume()
        print(json.dumps(state, indent=2))
    if args.compact:
        mm.compact()
        print("compacted")
    if not any([args.print_latest, args.resume, args.compact]):
        ap.print_help()


if __name__ == "__main__":
    main()
