from pathlib import Path
from bot_trade.eval import walk_forward


def test_walk_forward(tmp_path):
    log = tmp_path
    log.joinpath('reward.log').write_text('0,0.1\n1,0.2\n2,-0.1\n3,0.05\n', encoding='utf-8')
    res = walk_forward.walk_forward_eval(log, n_splits=3, embargo=0.05)
    assert 0.0 <= res['pass_ratio'] <= 1.0
    assert len(res['folds']) > 0
