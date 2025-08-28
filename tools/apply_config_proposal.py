import argparse
import json
import os
from typing import Any, Dict, List

import yaml

PROPOSAL_FILE = os.path.join('config', 'config.proposals.yaml')
CONFIG_FILE = os.path.join('config', 'config.yaml')


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def merge_cfg(cfg: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def apply(proposals: List[Dict[str, Any]], apply: bool = False) -> Dict[str, Any]:
    cfg = load_yaml(CONFIG_FILE)
    for prop in proposals:
        merge_cfg(cfg, prop['path'], prop['value'])
    if apply:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(cfg, fh)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--proposals', default=PROPOSAL_FILE)
    args = ap.parse_args()
    proposals_cfg = load_yaml(args.proposals)
    proposals = proposals_cfg.get('proposals') if isinstance(proposals_cfg, dict) else proposals_cfg
    if not proposals:
        print('no proposals')
        return
    new_cfg = apply(proposals, apply=args.apply)
    print(json.dumps(new_cfg, indent=2))


if __name__ == '__main__':
    main()
