import argparse
import json
import os
from typing import Dict

RUN_STATE_PATH = os.path.join('memory', 'run_state.json')


def load_state(path: str = RUN_STATE_PATH) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fh:
        try:
            return json.load(fh)
        except Exception:
            return {}


def save_state(state: Dict, path: str = RUN_STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
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
