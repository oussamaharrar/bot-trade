import pytest

from bot_trade.eval.gates import threshold_gate


def test_gate_pass_and_fail(tmp_path):
    metrics = {"sharpe": 1.0, "sortino": 0.7, "max_drawdown": 0.1, "win_rate": 0.6}
    thresholds = {"min_sharpe": 0.5, "min_sortino": 0.5, "max_drawdown": 0.2, "min_winrate": 0.5}
    res = threshold_gate(metrics, thresholds, promote_if=True, promotion_path=tmp_path / "promotion.json")
    assert res.passed and res.pass_ratio == 1.0 and res.promote_if
    assert (tmp_path / "promotion.json").exists()

    metrics2 = {"sharpe": 0.2, "sortino": 0.1, "max_drawdown": 0.3, "win_rate": 0.4}
    res2 = threshold_gate(metrics2, thresholds)
    assert not res2.passed
    assert set(res2.reasons) == {"min_sharpe", "min_sortino", "max_drawdown", "min_winrate"}
    assert res2.pass_ratio == 0.0

    with pytest.raises(SystemExit):
        threshold_gate(metrics2, thresholds, fail_hard=True)
