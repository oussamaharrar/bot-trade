import pytest
from bot_trade.ui import whitelist


def test_reject_non_whitelisted():
    with pytest.raises(whitelist.ValidationError):
        whitelist.build_command("train_rl", {"symbol":"BTCUSDT","frame":"1m","total_steps":1,"n_envs":1,"device":"cpu","foo":1})


def test_accept_training_and_eval():
    cmd = whitelist.build_command("train_rl", {
        "symbol":"BTCUSDT","frame":"1m","total_steps":1,"n_envs":1,"device":"cpu"})
    assert cmd[:3] == ["python","-m","bot_trade.train_rl"]
    cmd2 = whitelist.build_command("eval_run", {"symbol":"BTCUSDT","frame":"1m","run_id":"latest"})
    assert cmd2[:3] == ["python","-m","bot_trade.tools.eval_run"]
