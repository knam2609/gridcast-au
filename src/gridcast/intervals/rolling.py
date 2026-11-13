"""Order statistics of sliding windows of past scores, per lead and group (m3_plan.md §2.2).

The rolling conformal intervals need, at every issue time ``i`` and lead ``h``, the k-th
smallest calibration score among the pairs ``(j, h)`` that

* are **known** at ``i``: ``j ≤ i − lag[h]`` (the target of ``(j, h)`` has ended by issue
  time ``i``; the caller derives ``lag`` from the issue cadence);
* are **recent**: ``j > i − lag[h] − window``, i.e. the latest ``window`` issue times whose
  lead-``h`` target is known, so every lead has a window of the same length;
* are **valid** (the scored pairs of the common mask); and
* share the **group** of the target being forecast (e.g. its time-of-day block).

Scores are ranked once per (lead, group). Each window keeps a Fenwick (binary indexed)
tree over those ranks, updated by +1/−1 as pairs enter and leave, and a k-th-smallest
query is a binary descent of the tree: O(log n) per lead, vectorised over leads, windows
and queries. The caller advances issue time by issue time.

Ranking every score up front does not leak: a query returns the score of a pair inside the
window, whatever the ranks of the pairs outside it. The leakage property tests check this.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class RollingOrderStats:
    """Sliding windows of several lengths over one ``(n, H)`` score matrix.

    Parameters
    ----------
    scores:
        ``(n, H)`` calibration scores; only ``valid`` entries are used (they must be finite).
    valid:
        ``(n, H)`` bool, the pairs that may enter a calibration window.
    groups:
        ``(n, H)`` int group label of each pair's target, ``0 ≤ label < n_groups``.
    lag:
        ``(H,)`` issue steps after which a lead's target is known (≥ 1).
    windows:
        window lengths in issue steps (each ≥ 1).
    """

    def __init__(
        self,
        scores: np.ndarray,
        valid: np.ndarray,
        groups: np.ndarray,
        n_groups: int,
        lag: np.ndarray,
        windows: Sequence[int],
    ) -> None:
        n, n_leads = scores.shape
        if valid.shape != scores.shape or groups.shape != scores.shape:
            raise ValueError("scores, valid and groups must have the same shape")
        if lag.shape != (n_leads,) or np.any(lag < 1):
            raise ValueError("lag must hold one value >= 1 per lead")
        if not windows or min(windows) < 1:
            raise ValueError("windows must be positive")
        if np.any(valid & ~np.isfinite(scores)):
            raise ValueError("a valid pair has a non-finite score")
        if np.any(valid & ((groups < 0) | (groups >= n_groups))):
            raise ValueError("group labels must lie in [0, n_groups)")

        self.n = n
        self.n_leads = n_leads
        self.n_groups = n_groups
        self.lag = lag.astype(np.int64)
        self.windows = np.asarray(windows, dtype=np.int64)
        self._valid = valid
        self._groups = groups.astype(np.int16)
        self._leads = np.arange(n_leads)

        # Rank every valid score within its (lead, group), 1-based; ties keep issue order.
        rank = np.zeros((n, n_leads), dtype=np.int32)
        size = np.zeros((n_leads, n_groups), dtype=np.int64)
        for h in range(n_leads):
            size[h] = np.bincount(self._groups[valid[:, h], h], minlength=n_groups)
        # Flat layout: tree (and sorted values) of (h, g) occupy [off, off + size + 1);
        # position off + r holds rank r, position off is unused.
        off = np.zeros((n_leads, n_groups), dtype=np.int64)
        off.reshape(-1)[1:] = np.cumsum(size.reshape(-1) + 1)[:-1]
        total = int((size + 1).sum())
        vals = np.full(total, np.nan)
        for h in range(n_leads):
            idx = np.flatnonzero(valid[:, h])
            g = self._groups[idx, h]
            s = scores[idx, h].astype(np.float64)
            order = np.lexsort((idx, s, g))
            g_sorted = g[order]
            starts = np.cumsum(size[h]) - size[h]
            r = np.arange(len(order)) - starts[g_sorted] + 1
            rank[idx[order], h] = r
            vals[off[h, g_sorted] + r] = s[order]

        self._rank = rank
        self._size = size
        self._off = off
        self._vals = vals
        self._total = total
        self._tree = np.zeros((len(self.windows), total), dtype=np.int32)
        self._tree_flat = self._tree.reshape(-1)
        self._count = np.zeros((len(self.windows), n_leads, n_groups), dtype=np.int64)
        self._log = max(int(size.max()), 1).bit_length()
        self._next = 0

    # ------------------------------------------------------------------ updates

    def _update(self, rows: np.ndarray, leads: np.ndarray, pairs: np.ndarray, delta: int) -> None:
        """Add ``delta`` for pair ``pairs[e]`` of lead ``leads[e]`` in window ``rows[e]``.

        Events must be distinct (window, lead) combinations, so that no two events touch
        the same tree and fancy-indexed ``+=`` is exact.
        """
        g = self._groups[pairs, leads].astype(np.int64)
        pos = self._rank[pairs, leads].astype(np.int64)
        lim = self._size[leads, g]
        base = rows * self._total + self._off[leads, g]
        self._count[rows, leads, g] += delta
        while pos.size:
            self._tree_flat[base + pos] += delta
            pos = pos + (pos & -pos)
            keep = pos <= lim
            if not keep.all():
                pos, lim, base = pos[keep], lim[keep], base[keep]

    def advance(self, i: int) -> None:
        """Move every window to issue time ``i`` (calls must come in order 0, 1, 2, …)."""
        if i != self._next:
            raise ValueError(f"advance({i}) called out of order (expected {self._next})")
        n_win = len(self.windows)
        j_in = i - self.lag
        ok = (j_in >= 0) & (j_in < self.n)
        ok[ok] = self._valid[j_in[ok], self._leads[ok]]
        if ok.any():
            leads = np.tile(self._leads[ok], n_win)
            rows = np.repeat(np.arange(n_win), int(ok.sum()))
            self._update(rows, leads, np.tile(j_in[ok], n_win), +1)
        j_out = j_in[None, :] - self.windows[:, None]  # (W, H)
        ok_out = (j_out >= 0) & (j_out < self.n)
        rows, leads = np.nonzero(ok_out)
        pairs = j_out[rows, leads]
        keep = self._valid[pairs, leads]
        if keep.any():
            self._update(rows[keep], leads[keep], pairs[keep], -1)
        self._next = i + 1

    # ------------------------------------------------------------------ queries

    def counts(self, group: np.ndarray) -> np.ndarray:
        """``(W, H)`` number of pairs in each window that share each lead's ``group``."""
        return self._count[:, self._leads, group]

    def kth(self, k: np.ndarray, group: np.ndarray) -> np.ndarray:
        """k-th smallest score in each window's group ``group[h]`` of each lead.

        ``k`` is ``(W, Q, H)`` (1-based ranks; Q queries per window and lead). Returns
        ``(W, Q, H)`` float64, NaN where ``k`` is outside ``[1, count]``.
        """
        n_win = len(self.windows)
        if k.shape[0] != n_win or k.shape[2] != self.n_leads:
            raise ValueError("k must have shape (W, Q, H)")
        off = self._off[self._leads, group]  # (H,)
        size = self._size[self._leads, group]  # (H,)
        base = np.arange(n_win)[:, None, None] * self._total + off[None, None, :]
        k = k.astype(np.int64)
        pos = np.zeros(k.shape, dtype=np.int64)
        rem = k.copy()
        for p in range(self._log - 1, -1, -1):
            nxt = pos + (1 << p)
            ok = nxt <= size
            t = self._tree_flat[base + np.where(ok, nxt, 0)]
            ok &= t < rem
            pos = np.where(ok, nxt, pos)
            rem = np.where(ok, rem - t, rem)
        count = self.counts(group)[:, None, :]  # (W, 1, H)
        good = (k >= 1) & (k <= count)
        rank = np.minimum(pos + 1, size)
        out = self._vals[off + rank]
        return np.where(good, out, np.nan)
