import argparse
import json
import os
import uuid
from typing import Any, Dict, List

import numpy as np
import pandas as pd

BASE_DIR = os.path.join('memory', 'knowledge')
NODES_PATH = os.path.join(BASE_DIR, 'kg_nodes.parquet')
EDGES_PATH = os.path.join(BASE_DIR, 'kg_edges.parquet')
EMB_PATH = os.path.join(BASE_DIR, 'embeddings.npy')
EMB_META_PATH = os.path.join(BASE_DIR, 'meta.parquet')
SUMMARIES_PATH = os.path.join(BASE_DIR, 'summaries.jsonl')
TACTICS_PATH = os.path.join(BASE_DIR, 'tactics.jsonl')
PROPOSALS_PATH = os.path.join(BASE_DIR, 'config_proposals.jsonl')

EMB_DIM = 384


def _ensure_base() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)


def _load_df(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


def _append_df(path: str, row: Dict[str, Any]) -> None:
    df = _load_df(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(path, index=False)


def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    _ensure_base()
    tmp = path + '.tmp'
    lines = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            lines = fh.read().splitlines()
    lines.append(json.dumps(rec))
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + ('\n' if lines else ''))
    os.replace(tmp, path)


def _embed(text: str) -> np.ndarray:
    # placeholder embedding using hash-based seed for determinism
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=EMB_DIM).astype('float32')


def ingest_text(doc: str, meta: Dict[str, Any]) -> str:
    _ensure_base()
    node_id = str(uuid.uuid4())
    row = {
        'id': node_id,
        'type': meta.get('type', 'note'),
        'symbol': meta.get('symbol'),
        'frame': meta.get('frame'),
        'ts': meta.get('ts'),
        'title': meta.get('title'),
        'text': doc,
        'tags': meta.get('tags', []),
    }
    _append_df(NODES_PATH, row)
    emb = _embed(doc)
    if os.path.exists(EMB_PATH):
        arr = np.load(EMB_PATH)
        arr = np.vstack([arr, emb])
    else:
        arr = np.array([emb])
    np.save(EMB_PATH, arr)
    meta_row = { 'id': node_id }
    _append_df(EMB_META_PATH, meta_row)
    return node_id


def link(src: str, dst: str, typ: str, weight: float = 1.0) -> None:
    row = {
        'src_id': src,
        'dst_id': dst,
        'type': typ,
        'weight': weight,
    }
    _append_df(EDGES_PATH, row)


def summarize(symbol: str, frame: str, last_n: int = 20) -> str:
    df = _load_df(NODES_PATH)
    df = df[(df.symbol == symbol) & (df.frame == frame)]
    df = df.tail(last_n)
    text = '\n'.join(df.text.astype(str).tolist())
    rec = {
        'symbol': symbol,
        'frame': frame,
        'summary': text[:1000],
    }
    _append_jsonl(SUMMARIES_PATH, rec)
    return text


def propose_config_edits(current_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    lr = current_cfg.get('rl', {}).get('learning_rate')
    try:
        lr_val = float(lr) if lr is not None else None
    except Exception:
        lr_val = None
    if lr_val and lr_val > 0.0005:
        proposals.append({'path': ['rl', 'learning_rate'], 'value': 0.0005})
    if proposals:
        _append_jsonl(PROPOSALS_PATH, {'ts': uuid.uuid4().hex, 'proposals': proposals})
    return proposals


def search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    if not os.path.exists(EMB_PATH):
        return []
    emb = _embed(query)
    arr = np.load(EMB_PATH)
    meta = _load_df(EMB_META_PATH)
    if arr.shape[0] != len(meta):
        return []
    sims = arr @ emb / (np.linalg.norm(arr, axis=1) * np.linalg.norm(emb) + 1e-9)
    idx = np.argsort(-sims)[:k]
    nodes = _load_df(NODES_PATH)
    out = []
    for i in idx:
        node_id = meta.iloc[i]['id']
        rec = nodes[nodes.id == node_id].iloc[0].to_dict()
        if isinstance(rec.get('tags'), np.ndarray):
            rec['tags'] = rec['tags'].tolist()
        rec['score'] = float(sims[i])
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--search')
    ap.add_argument('--ingest')
    ap.add_argument('--meta')
    args = ap.parse_args()
    if args.ingest:
        meta = json.loads(args.meta or '{}')
        node_id = ingest_text(args.ingest, meta)
        print(node_id)
    elif args.search:
        res = search(args.search)
        print(json.dumps(res, indent=2))
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
