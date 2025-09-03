import argparse
import json
from pathlib import Path
from typing import List

from PIL import Image

from bot_trade.config.rl_paths import DEFAULT_KB_FILE, RunPaths, memory_dir


def _load_kb_entry(run_id: str | None) -> tuple[str | None, dict | None]:
    kb = Path(DEFAULT_KB_FILE)
    if not kb.exists():
        return None, None
    entry = None
    try:
        lines = kb.read_text(encoding="utf-8").splitlines()
        if run_id is None:
            for ln in reversed(lines):
                if ln.strip():
                    entry = json.loads(ln)
                    break
        else:
            for ln in lines:
                obj = json.loads(ln)
                if obj.get("run_id") == run_id:
                    entry = obj
                    break
    except Exception:
        return None, None
    if not entry:
        return None, None
    return entry.get("run_id"), entry


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Development data checks")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--run-id", default="latest")
    args = ap.parse_args(argv)

    details: list[str] = []

    parquet_store = Path("data_store") / "parquet"
    if not parquet_store.exists():
        details.append("missing_parquet")

    rid_arg = None if str(args.run_id).lower() in {"latest", "last"} else args.run_id
    run_id, kb_entry = _load_kb_entry(rid_arg)
    if not run_id or not kb_entry:
        print("[LATEST] none")
        return 2

    algo = (kb_entry.get("algorithm") or "").upper()
    if not algo:
        details.append("missing_algorithm")
    if kb_entry.get("eval", {}).get("max_drawdown") is None:
        details.append("missing_eval_max_drawdown")

    rp = RunPaths(args.symbol, args.frame, run_id, algo or "PPO")
    charts = rp.reports / "charts"

    def _check_png(name: str, tag: str) -> None:
        p = charts / name
        ok = p.exists()
        size = dpi = 0
        if ok:
            size = p.stat().st_size
            try:
                with Image.open(p) as im:
                    info = im.info.get("dpi") or (0, 0)
                    dpi = int(round(info[0] if isinstance(info, tuple) else info))
            except Exception:
                dpi = 0
        if (not ok) or size < 1024 or dpi < 120:
            details.append(f"{tag}_png")

    _check_png("risk_flags.png", "risk_flags")
    _check_png("regimes.png", "regimes")

    ai_core_used = kb_entry.get("ai_core", {}).get("signals_count", 0) > 0
    if ai_core_used:
        sig = memory_dir() / "Knowlogy" / "signals.jsonl"
        try:
            data = sig.read_bytes()
            if not data.endswith(b"\n"):
                with sig.open("ab") as fh:
                    fh.write(b"\n")
                print(f"[IO] fixed_trailing_newline file={sig}")
        except FileNotFoundError:
            details.append("signals_jsonl_missing")
        except Exception:
            details.append("signals_jsonl_missing")

    status = "ok" if not details else "warnings"
    print(f"[CHECKS] status={status} count={len(details)} details=[{','.join(details)}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    from bot_trade.tools.encoding import force_utf8

    force_utf8()
    raise SystemExit(main())

