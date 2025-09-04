import pytest

from bot_trade.tools import ui_panel, whitelist


def test_registry_renders_and_filters():
    model = ui_panel.PanelModel()
    params = {"symbol": "BTCUSDT", "frame": "1m"}
    cmd = model.build_train_command(params)
    assert cmd[:3] == ["python", "-m", "bot_trade.train_rl"]
    with pytest.raises(whitelist.ValidationError):
        whitelist.build_command("train", params, ["--bad-flag"])
