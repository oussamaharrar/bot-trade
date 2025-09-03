from __future__ import annotations

"""Data splitting utilities (walk-forward and purged K-Fold)."""

from typing import Iterator, Sequence, Tuple, List


def walk_forward(indices: Sequence[int], train: int, test: int) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
    """Yield walk-forward splits for ordered ``indices``."""
    n = len(indices)
    step = test
    for start in range(0, n - train - test + 1, step):
        yield indices[start : start + train], indices[start + train : start + train + test]


def purged_kfold(n: int, k: int, embargo: int) -> Iterator[Tuple[List[int], List[int]]]:
    """Yield Purged/Embargoed K-Fold splits over ``n`` observations."""
    fold = n // k
    for i in range(k):
        test_start = i * fold
        test_end = test_start + fold
        train_left = list(range(0, max(0, test_start - embargo)))
        train_right = list(range(min(n, test_end + embargo), n))
        yield train_left + train_right, list(range(test_start, test_end))
