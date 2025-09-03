from bot_trade.config.encoding import force_utf8
import os
import locale


def test_force_utf8(monkeypatch, capsys):
    monkeypatch.setattr(locale, "getpreferredencoding", lambda _: "cp1252")
    force_utf8()
    assert os.environ.get("PYTHONIOENCODING") == "utf-8"
    out = capsys.readouterr().out
    assert "[ENCODING] forced UTF-8" in out

