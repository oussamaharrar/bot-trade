from bot_trade.config.encoding import force_utf8
import os


def test_force_utf8(capsys):
    force_utf8()
    assert os.environ.get("PYTHONIOENCODING") == "utf-8"
    out = capsys.readouterr().out.strip()
    assert out.startswith("[ENCODING] PYTHONIOENCODING=")

