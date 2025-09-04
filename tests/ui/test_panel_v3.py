import sys
import time
import pytest
from bot_trade.tools import ui_panel, whitelist, runctx


def test_registry_and_queue(tmp_path):
    model = ui_panel.PanelModel()
    params = {"symbol": "BTCUSDT", "frame": "1m"}
    cmd = model.build_train_command(params)
    assert cmd[0] == "python"
    with pytest.raises(whitelist.ValidationError):
        whitelist.build_command("train", params, ["--bad-flag"])

    script = [sys.executable, "-c", "import time; time.sleep(1)"]
    h1 = runctx.start(script, tee_path=tmp_path/"a.log")
    h2 = runctx.start(script, tee_path=tmp_path/"b.log")
    time.sleep(0.2)
    assert runctx.tail(h1.id) == "" or isinstance(runctx.tail(h1.id), str)
    runctx.stop(h1.id)
    runctx.stop(h2.id)

    model.handle_log_line("[DEBUG_EXPORT] foo=1")
    model.handle_log_line("[CHARTS] dir=/tmp/charts images=6")
    model.handle_log_line("[POSTRUN] run_id=abcd1234 symbol=BTCUSDT frame=1m algorithm=PPO eval_sharpe=1.1 eval_max_drawdown=-0.12")
    assert model.last_run_id == "abcd1234"
    assert model.last_postrun.get("eval_sharpe") == 1.1
