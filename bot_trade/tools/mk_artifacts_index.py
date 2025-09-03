from __future__ import annotations

import argparse
import datetime as dt

from .atomic_io import write_json
from .latest import latest_run
from .paths import report_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser("mk_artifacts_index")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--algorithm")
    ns = ap.parse_args(argv)

    algo = ns.algorithm
    run_id = ns.run_id
    base = report_dir(ns.symbol, ns.frame, algo=algo)
    if run_id == "latest":
        rid = latest_run(ns.symbol, ns.frame, base.parent.parent)
        if not rid:
            print("[LATEST] none")
            return
        run_id = rid
    run_dir = base / run_id

    charts_dir = run_dir / "charts"
    charts = sorted(p.name for p in charts_dir.glob("*.png"))

    data = {
        "run_id": run_id,
        "algorithm": algo or "",
        "symbol": ns.symbol,
        "frame": ns.frame,
        "charts": charts,
        "tearsheet": "tearsheet.html",
        "kb_line_path": "memory/Knowlogy/kb.jsonl",
        "created_utc": dt.datetime.utcnow().isoformat(),
    }
    out_path = run_dir / "artifacts" / "index.json"
    write_json(out_path, data)
    if algo:
        legacy = report_dir(ns.symbol, ns.frame) / run_id / "artifacts" / "index.json"
        write_json(legacy, data)
    print(f"[ARTIFACTS] index={out_path.resolve()} charts={len(charts)}")


if __name__ == "__main__":
    main()
