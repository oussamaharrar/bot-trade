#!/usr/bin/env python3
import argparse, yaml, os, sys

def deep_set(d, path, value):
    cur = d
    *keys, last = path.split(".")
    for k in keys:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    if isinstance(value, str):
        if value.lower() in ("true","false"):
            v = value.lower()=="true"
        else:
            try:
                v = float(value) if "." in value else int(value)
            except:
                v = value
    else:
        v = value
    cur[last] = v

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="config/config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="key.path=value")
    args = ap.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for s in args.__dict__["set"]:
        if "=" not in s:
            print(f"Bad --set: {s}", file=sys.stderr); sys.exit(2)
        k, v = s.split("=", 1)
        deep_set(cfg, k.strip(), v.strip())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
