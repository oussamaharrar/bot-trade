import argparse
import json
import os
from typing import Dict, Iterator, Optional
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

from bot_trade.config.rl_paths import ensure_utf8, memory_dir

INDEX_PATH = memory_dir() / 'state_index.jsonl'


def _atomic_append(path: Path, line: str) -> None:
    tmp = path.with_suffix('.tmp')
    lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    lines.append(line)
    with ensure_utf8(tmp, csv_newline=False) as fh:
        fh.write('\n'.join(lines) + ('\n' if lines else ''))
    os.replace(tmp, path)


def update_index(token: Dict) -> None:
    line = json.dumps(token, sort_keys=True)
    _atomic_append(INDEX_PATH, line)


def get_resume_token(artifact: str, symbol: str, frame: str) -> Optional[Dict]:
    if not INDEX_PATH.exists():
        return None
    for line in reversed(INDEX_PATH.read_text(encoding='utf-8').splitlines()):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get('artifact') == artifact and rec.get('symbol') == symbol and rec.get('frame') == frame:
            return rec
    return None


def tail_from_offset(path: Path, byte_offset: Optional[int] = None, line_offset: Optional[int] = None) -> Iterator[str]:
    with path.open('r', encoding='utf-8') as fh:
        if byte_offset is not None:
            fh.seek(max(0, byte_offset))
        elif line_offset is not None:
            for _ in range(max(0, line_offset)):
                fh.readline()
        while True:
            line = fh.readline()
            if not line:
                break
            yield line.rstrip('\n')


def _cli_show(symbol: Optional[str], frame: Optional[str]):
    if not INDEX_PATH.exists():
        print('no index found')
        return
    with INDEX_PATH.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if symbol and rec.get('symbol') != symbol:
                continue
            if frame and rec.get('frame') != frame:
                continue
            print(json.dumps(rec, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--symbol')
    ap.add_argument('--frame')
    args = ap.parse_args()
    if args.show:
        _cli_show(args.symbol, args.frame)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
