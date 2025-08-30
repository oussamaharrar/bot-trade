import argparse
import json
import os
from pathlib import Path
from typing import Dict

from bot_trade.config.rl_paths import ensure_utf8, memory_dir

RUN_STATE_PATH = memory_dir() / 'run_state.json'


def load_state(path: Path = RUN_STATE_PATH) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_state(state: Dict, path: Path = RUN_STATE_PATH) -> None:
    tmp = path.with_suffix('.tmp')
    with ensure_utf8(tmp, csv_newline=False) as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--symbol')
    ap.add_argument('--frame')
    args = ap.parse_args()
    if args.resume:
        st = load_state()
        if st:
            print(json.dumps(st, indent=2))
        else:
            print('{}')
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
