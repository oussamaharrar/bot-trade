from pathlib import Path
from bot_trade.eval import tearsheet


class DummyRP:
    def __init__(self, base: Path):
        self.performance_dir = base
        self.logs = base


def test_tearsheet_generation(tmp_path, capsys):
    base = tmp_path
    base.joinpath('summary.json').write_text('{}', encoding='utf-8')
    base.joinpath('metrics.csv').write_text('sharpe\n', encoding='utf-8')
    base.joinpath('reward.log').write_text('0,0.1\n1,0.2\n', encoding='utf-8')
    rp = DummyRP(base)
    out = tearsheet.generate_tearsheet(rp)
    captured = capsys.readouterr().out
    assert '[TEARSHEET] out=' in captured
    assert out.exists()
