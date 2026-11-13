"""RollingOrderStats against a brute-force sort of each window (m3_plan.md §2.2)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gridcast.intervals.rolling import RollingOrderStats


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_kth_matches_sorted_window(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, n_leads, n_groups = (
        int(rng.integers(2, 60)),
        int(rng.integers(1, 5)),
        int(rng.integers(1, 4)),
    )
    scores = np.round(rng.normal(size=(n, n_leads)), 1)  # ties on purpose
    valid = rng.random((n, n_leads)) > 0.2
    scores[~valid & (rng.random((n, n_leads)) > 0.5)] = np.nan
    groups = rng.integers(0, n_groups, (n, n_leads))
    lag = rng.integers(1, 4, n_leads)
    windows = sorted({int(w) for w in rng.integers(1, 20, 3)})
    eng = RollingOrderStats(scores, valid, groups, n_groups, lag, windows)
    for i in range(n):
        eng.advance(i)
        k = rng.integers(0, 12, (len(windows), 3, n_leads))
        got = eng.kth(k, groups[i])
        counts = eng.counts(groups[i])
        for w, length in enumerate(windows):
            for h in range(n_leads):
                last = i - lag[h]
                pool = [
                    j
                    for j in range(max(0, last - length + 1), last + 1)
                    if valid[j, h] and groups[j, h] == groups[i, h]
                ]
                ordered = np.sort(scores[pool, h])
                assert counts[w, h] == len(pool)
                for q in range(3):
                    kk = k[w, q, h]
                    if 1 <= kk <= len(ordered):
                        assert got[w, q, h] == ordered[kk - 1]
                    else:
                        assert np.isnan(got[w, q, h])


def test_rejects_bad_inputs() -> None:
    s = np.zeros((4, 2))
    ok = np.ones((4, 2), bool)
    g = np.zeros((4, 2), int)
    with pytest.raises(ValueError, match="lag"):
        RollingOrderStats(s, ok, g, 1, np.array([0, 1]), [2])
    with pytest.raises(ValueError, match="non-finite"):
        RollingOrderStats(np.full((4, 2), np.nan), ok, g, 1, np.array([1, 1]), [2])
    with pytest.raises(ValueError, match="group"):
        RollingOrderStats(s, ok, g + 1, 1, np.array([1, 1]), [2])
    eng = RollingOrderStats(s, ok, g, 1, np.array([1, 1]), [2])
    with pytest.raises(ValueError, match="out of order"):
        eng.advance(1)
