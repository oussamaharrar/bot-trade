import pytest

from bot_trade.ui.whitelist import build_command, load_whitelist


def test_reject_non_whitelisted():
    wl = load_whitelist()
    with pytest.raises(ValueError):
        build_command("nonexistent", {}, wl)


def test_template_expansion():
    wl = {"demo": ["echo", "{symbol}", "{frame}"]}
    cmd = build_command("demo", {"symbol": "BTC", "frame": "1h"}, wl)
    assert cmd == ["echo", "BTC", "1h"]
