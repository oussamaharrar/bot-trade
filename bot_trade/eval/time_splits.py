"""Chronological data split helpers used for walk-forward analysis."""

from collections.abc import Iterable, Iterator, Sequence


def walk_forward(
    data: Sequence[int] | Iterable[int], train: int, test: int, step: int | None = None
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield expanding walk-forward train/test indices.

    Parameters
    ----------
    data: sequence of indices (only length is used).
    train: number of samples in the training window.
    test: number of samples in the test window.
    step: step size between windows; defaults to ``test``.
    """

    idx = list(data)
    n = len(idx)
    step = step or test
    for start in range(0, n - train - test + 1, step):
        t0 = start
        t1 = start + train
        train_idx = idx[t0:t1]
        test_idx = idx[t1:t1 + test]
        if not test_idx:
            break
        yield train_idx, test_idx


def purged_kfold(n_samples: int, n_splits: int, embargo: float = 0.0) -> Iterator[tuple[list[int], list[int]]]:
    """Yield purged K-fold splits to avoid leakage.

    Parameters
    ----------
    n_samples: total number of samples.
    n_splits: number of folds.
    embargo: fraction (0-1) of samples to embargo on each side of the test
        window.
    """

    idx = list(range(n_samples))
    fold_size = n_samples // n_splits
    embargo_n = int(embargo * n_samples)
    for i in range(n_splits):
        start = i * fold_size
        end = min(n_samples, start + fold_size)
        test_idx = idx[start:end]
        train_idx = [j for j in idx if j < start - embargo_n or j >= end + embargo_n]
        yield train_idx, test_idx
