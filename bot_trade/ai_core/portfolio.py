import json
import os
from typing import Any, Dict, List

STATE_FILE_TMPL = os.path.join('results', '{symbol}', '{frame}', 'portfolio_state.json')


def _state_path(symbol: str, frame: str) -> str:
    return STATE_FILE_TMPL.format(symbol=symbol, frame=frame)


def load_state(symbol: str, frame: str) -> Dict[str, Any]:
    path = _state_path(symbol, frame)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fh:
        try:
            return json.load(fh)
        except Exception:
            return {}


def save_state(symbol: str, frame: str, state: Dict[str, Any]) -> None:
    path = _state_path(symbol, frame)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def reset_with_balance(symbol: str, frame: str, balance: float) -> Dict[str, Any]:
    state = {
        'balance_start': balance,
        'balance': balance,
        'equity': balance,
        'positions': [],
        'pnl_cum': 0.0,
        'fees_cum': 0.0,
        'max_drawdown': 0.0,
        'vol_rolling': 0.0,
        'last_update_step': 0,
    }
    save_state(symbol, frame, state)
    return state


def apply_trade(state: Dict[str, Any], fill: Dict[str, Any]) -> None:
    qty = float(fill.get('qty', 0.0))
    price = float(fill.get('price', 0.0))
    fee = float(fill.get('fee', 0.0))
    side = fill.get('side', 'long')
    pnl = qty * price * (1 if side == 'long' else -1)
    state['balance'] = state.get('balance', 0.0) - pnl - fee
    state['fees_cum'] = state.get('fees_cum', 0.0) + fee
    state.setdefault('positions', []).append({
        'id': fill.get('id'),
        'symbol': fill.get('symbol'),
        'side': side,
        'qty': qty,
        'entry': price,
    })


def apply_mark_to_market(state: Dict[str, Any], price: float) -> None:
    equity = state.get('balance', 0.0)
    for pos in state.get('positions', []):
        direction = 1 if pos.get('side') == 'long' else -1
        equity += direction * pos.get('qty', 0.0) * (price - pos.get('entry', price))
    state['equity'] = equity
    peak = state.setdefault('equity_peak', equity)
    if equity > peak:
        state['equity_peak'] = equity
    dd = (peak - equity) / peak if peak else 0.0
    state['max_drawdown'] = max(state.get('max_drawdown', 0.0), dd)

