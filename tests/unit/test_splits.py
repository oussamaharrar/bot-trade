from bot_trade.data.splits import walk_forward, purged_kfold


def test_walk_forward():
    idx = list(range(10))
    splits = list(walk_forward(idx, train=4, test=2))
    assert splits[0][0] == [0, 1, 2, 3]
    assert splits[0][1] == [4, 5]


def test_purged_kfold():
    splits = list(purged_kfold(10, 5, 1))
    train0, test0 = splits[0]
    assert test0 == [0, 1]
    assert 0 not in train0 and 1 not in train0
