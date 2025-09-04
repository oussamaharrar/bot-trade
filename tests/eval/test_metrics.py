import pandas as pd
import pytest
from bot_trade.eval import metrics

def test_basic_metrics():
    returns = pd.Series([0.2, -0.1, 0.3, -0.05])
    assert metrics.sharpe(returns) > 0
    assert metrics.sortino(returns) > 0
    eq = metrics.to_equity_from_returns(returns, start=1.0)
    assert metrics.max_drawdown(eq) >= 0
    assert metrics.calmar(eq) is not None

def test_safe_nan():
    assert metrics.sharpe([]) is None
    assert metrics.sortino([]) is None
    assert metrics.calmar([]) is None
    assert metrics.max_drawdown([]) == 0.0
