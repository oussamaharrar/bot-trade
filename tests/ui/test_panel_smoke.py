import pytest
from bot_trade.tools import panel_gui, whitelist


def test_panel_model_build_and_parse(tmp_path):
    model = panel_gui.PanelModel()
    params = {
        "symbol": "BTCUSDT",
        "frame": "1m",
        "total_steps": "1",
        "n_envs": "1",
        "device": "cpu",
        "data_dir": str(tmp_path),
    }
    cmd = model.build_train_command(params)
    assert cmd[:3] == ["python", "-m", "bot_trade.train_rl"]
    with pytest.raises(whitelist.ValidationError):
        whitelist.build_command("train", {**params, "bad": 1})
    log = tmp_path / "log.txt"
    log.write_text(
        "[DEBUG_EXPORT] reward_rows=1 step_rows=1 train_rows=1 risk_rows=0 callbacks_rows=0 signals_rows=0\n"
        "[CHARTS] dir=/tmp/charts images=5\n"
        "[POSTRUN] run_id=abcd1234 symbol=BTCUSDT frame=1m algorithm=PPO eval_win_rate=0.5 eval_sharpe=1.2 eval_max_drawdown=-0.10\n",
        encoding="utf-8",
    )
    for line in log.read_text(encoding="utf-8").splitlines():
        model.handle_log_line(line)
    assert model.last_run_id == "abcd1234"
    assert model.last_postrun.get("eval_sharpe") == 1.2
