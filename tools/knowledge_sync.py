import argparse
import json
import os
from pathlib import Path


def collect_runs(results_dir: str, agents_dir: str) -> dict:
    data = {}
    for sym_dir in Path(results_dir).glob('*'):
        if not sym_dir.is_dir():
            continue
        sym = sym_dir.name
        for frame_dir in sym_dir.glob('*'):
            if not frame_dir.is_dir():
                continue
            frame = frame_dir.name
            key = f"{sym}:{frame}"
            info = {}
            train_csv = frame_dir / 'train_log.csv'
            if train_csv.exists():
                info['has_train'] = True
            eval_csv = frame_dir / 'evaluation.csv'
            if eval_csv.exists():
                info['has_eval'] = True
            best_model = Path(agents_dir) / sym / frame / 'deep_rl_best.zip'
            info['has_best_model'] = best_model.exists()
            data[key] = info
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--agents-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    summary = collect_runs(args.results_dir, args.agents_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
