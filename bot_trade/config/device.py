"""Device normalization utilities."""
from __future__ import annotations
from typing import Optional, Dict

def normalize_device(arg: str | None, env: Dict[str, str]) -> str | None:
    """Normalize device input to 'cpu', 'cuda:<idx>' or None for auto.

    Rules:
      - If env['CUDA_VISIBLE_DEVICES'] exists and is '' (empty string), force 'cpu'.
      - arg in {'cpu', 'CUDA', 'cuda'} -> 'cpu' or 'cuda:0' respectively.
      - 'cuda:<n>' stays as is if n is digit.
      - '<int>' -> 'cuda:<int>' if >=0 else 'cpu'
      - '-1' -> 'cpu'
      - None -> defer to auto after importing torch.
    """
    if env.get('CUDA_VISIBLE_DEVICES', None) == '':
        return 'cpu'
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        return None
    low = s.lower()
    if low in {'cpu', '-1'}:
        return 'cpu'
    if low in {'cuda', 'gpu'}:
        return 'cuda:0'
    if low.startswith('cuda:'):
        idx = low.split(':', 1)[1]
        return f'cuda:{idx}' if idx.isdigit() else 'cuda:0'
    try:
        idx = int(low)
        return 'cpu' if idx < 0 else f'cuda:{idx}'
    except ValueError:
        return s
