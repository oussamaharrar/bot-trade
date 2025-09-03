import os

from bot_trade.tools.force_utf8 import force_utf8


def test_force_utf8_once(capsys):
    force_utf8()
    force_utf8()
    assert os.environ.get("PYTHONIOENCODING") == "UTF-8"
    out = capsys.readouterr().out.splitlines()
    assert out == ["[ENCODING] UTF-8"]
