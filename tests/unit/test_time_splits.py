from bot_trade.eval.time_splits import purged_kfold, walk_forward


def test_walk_forward():
    idx = list(range(10))
    splits = list(walk_forward(idx, train=4, test=2))
    assert splits[0][0] == [0, 1, 2, 3]
    assert splits[0][1] == [4, 5]


def test_purged_kfold_deterministic_and_purged():
    splits = list(purged_kfold(10, 5, 0.1))
    train0, test0 = splits[0]
    assert test0 == [0, 1]
    # embargo removes index 2
    assert 0 not in train0 and 1 not in train0 and 2 not in train0
    # determinism
    splits2 = list(purged_kfold(10, 5, 0.1))
    assert splits == splits2
