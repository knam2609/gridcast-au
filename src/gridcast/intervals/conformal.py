"""Rolling conformal intervals with ACI and per-retrain selection (m3_plan.md §2.2).

Terms
-----
* A **pair** ``(i, h)`` is issue time ``i`` (an index into a regular issue grid) and lead
  ``h`` (1…H). Its target is the interval ending ``h`` steps after issue time ``i``. A step
  is 5 minutes (the dispatch interval), or 30 minutes for the half-hour periods of the AEMO
  comparison (:mod:`gridcast.intervals.halfhour`); ``ConformalProblem.step_minutes`` says.
* A **tail** is one side of one level's interval. Its bound is ``base ± Q``, where Q is a
  conformal quantile of the tail's past **scores**:

  - conformal around a point forecast ŷ: upper score ``y − ŷ``, lower score ``ŷ − y``,
    both tails based on ŷ (one signed-residual matrix serves both);
  - CQR: upper score ``y − q̂_hi``, lower score ``q̂_lo − y``, based on q̂_hi and q̂_lo.

  Separate quantiles for the two tails make intervals asymmetric, so a biased forecast
  gets a shifted interval rather than only a wider one.
* **Conformal quantile:** the k-th smallest of the window's n scores, with
  ``k = ⌈(n + 1)(1 − β)⌉`` and β the tail's miscoverage target (α/2 without ACI). ``k``
  is clipped to ``[1, n]``: a level beyond the window uses its most extreme score. Such
  clipped bounds are counted.
* **Information rule:** the window of ``(i, h)`` holds valid pairs ``(j, h)`` whose target
  is known at issue time ``i`` (``j ≤ i − lag(h)``), the latest ``window`` of them
  (:class:`~gridcast.intervals.rolling.RollingOrderStats`). With grouping ``lead_tod`` only
  pairs whose target shares the time-of-day block of ``(i, h)``'s target are used.
* **ACI** (adaptive conformal inference, Gibbs & Candès 2021), per lead, level and tail:
  ``β ← β + γ (α/2 − miss)``, applied at issue time ``j + lag(h)``, when the outcome of
  ``(j, h)`` becomes known. β is kept in ``[0, ½]``. With ``γ = 0``, β stays α/2.
* **Candidate stream:** a (window length, grouping, γ) combination. Every candidate runs
  on every issue time from the start of the data, each on past data only.
* **Selection:** at each retrain R, for every level and lead band, the candidate with the
  lowest mean interval score on the selection slice (pairs issued in the last
  ``selection_weeks`` before R whose target is known by R, like M2's early-stopping slice)
  gives the intervals of every issue time in ``[R, next R)``. Candidates missing a bound
  on the slice are not eligible; ties go to the first in grid order (window, grouping, γ).

All bookkeeping happens when information arrives: an outcome is used (for ACI, selection
and diagnostics) only at the issue time it becomes known, so intervals issued at ``i``
depend on nothing after ``i``. The leakage property tests check this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gridcast.intervals.rolling import RollingOrderStats
from gridcast.intervals.scores import alpha
from gridcast.logs import get_logger

log = get_logger(__name__)

TAILS: tuple[str, ...] = ("lo", "hi")
GROUPINGS: tuple[str, ...] = ("lead", "lead_tod")
STEP_MINUTES = 5
DIAGNOSTIC_STATS: tuple[str, ...] = ("n", "covered", "below", "above", "width", "interval_score")
BETA_MAX = 0.5


# --------------------------------------------------------------------------- definitions


@dataclass(frozen=True, slots=True)
class Candidate:
    window_weeks: float
    grouping: str
    gamma: float

    @property
    def key(self) -> str:
        return f"w{self.window_weeks:g}_{self.grouping}_g{self.gamma:g}"


@dataclass(frozen=True, slots=True)
class CandidateGrid:
    """The product of window lengths, groupings and ACI steps, in that (grid) order."""

    window_weeks: tuple[float, ...]
    groupings: tuple[str, ...]
    gammas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (self.window_weeks and self.groupings and self.gammas):
            raise ValueError("every grid axis needs at least one value")
        if unknown := set(self.groupings) - set(GROUPINGS):
            raise ValueError(f"unknown groupings {sorted(unknown)}")
        if any(g < 0 for g in self.gammas):
            raise ValueError("ACI steps must be >= 0")
        for axis in (self.window_weeks, self.groupings, self.gammas):
            if len(set(axis)) != len(axis):
                raise ValueError("grid values must be distinct")

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.window_weeks), len(self.groupings), len(self.gammas)

    def candidates(self) -> list[Candidate]:
        return [
            Candidate(w, g, gm)
            for w in self.window_weeks
            for g in self.groupings
            for gm in self.gammas
        ]


@dataclass(frozen=True, slots=True)
class Tail:
    """One side of one level: which score matrix it reads, and its base forecast.

    ``negate`` means the score matrix holds minus this tail's score (so the two tails of
    conformal-on-point share one signed-residual matrix).
    """

    score: int
    base: np.ndarray  # (n, H)
    negate: bool = False


@dataclass(slots=True)
class ConformalProblem:
    """Everything the rolling conformal run reads, on a regular issue grid.

    ``y`` holds the actuals (NaN where missing); ``valid`` the pairs that are scored and
    may calibrate (the common mask). ``groups`` maps a grouping name to the target's group
    label per pair and the number of groups (``lead`` is a single group).
    """

    origin_times: pd.DatetimeIndex
    issue_every_steps: int
    latency_steps: int
    y: np.ndarray
    valid: np.ndarray
    scores: list[np.ndarray]
    tails: dict[tuple[int, str], Tail]
    groups: dict[str, tuple[np.ndarray, int]]
    step_minutes: int = STEP_MINUTES  # length of one lead step

    @property
    def shape(self) -> tuple[int, int]:
        n, n_leads = self.y.shape
        return n, n_leads


@dataclass(frozen=True, slots=True)
class Selection:
    retrain: str  # ISO timestamp (UTC)
    level: int
    band: str
    chosen: str  # candidate key
    n_pairs: int
    scores: dict[str, float]  # candidate key -> mean interval score (NaN if not eligible)


@dataclass(slots=True)
class ConformalResult:
    origin_times: pd.DatetimeIndex  # issue times with intervals (from the first retrain)
    lower: dict[int, np.ndarray]  # level -> (n_out, H) float32
    upper: dict[int, np.ndarray]
    candidates: list[Candidate]
    selections: list[Selection]
    daily: pd.DataFrame  # per NEM day of issue, band, candidate and level (DIAGNOSTIC_STATS)
    clipped: pd.DataFrame  # per candidate, level and tail: bounds at the window's extreme
    crossed: dict[int, int] = field(default_factory=dict)  # level -> empty intervals output


def known_lag(horizons: np.ndarray, issue_every_steps: int, latency_steps: int = 0) -> np.ndarray:
    """Issue steps after which each lead's target is known.

    The target of ``(j, h)`` ends ``h`` steps after issue time ``j`` and is known at issue
    time ``i`` when it ends at least ``latency_steps`` before it:
    ``(i − j) · issue_every_steps ≥ h + latency_steps``.
    """
    return np.asarray(-(-(horizons + latency_steps) // issue_every_steps), dtype=np.int64)


def conformal_rank(count: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank ``k = ⌈(n + 1)(1 − β)⌉`` clipped to ``[1, n]``, and whether it was clipped.

    A tolerance of 1e-9 keeps exact products (10 × 0.9) from rounding up.
    """
    raw = np.ceil((count + 1) * (1.0 - beta) - 1e-9)
    k = np.clip(raw, 1, np.maximum(count, 1)).astype(np.int64)
    return k, (raw > count) | (raw < 1)


def window_steps(weeks: float, issue_minutes: int) -> int:
    steps = weeks * 7 * 1440 / issue_minutes
    if steps < 1 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"a {weeks}-week window is not a whole number of issue times")
    return round(steps)


def epoch_minutes(times: pd.DatetimeIndex | Sequence[pd.Timestamp]) -> np.ndarray:
    idx = pd.DatetimeIndex(list(times))
    return ((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(minutes=1)).to_numpy(np.int64)


def band_index(bands: Sequence[tuple[int, int]], n_leads: int) -> np.ndarray:
    """Band of each lead (0-based); bands must tile 1…H in order."""
    expected = 1
    out = np.empty(n_leads, dtype=np.int64)
    for b, (lo, hi) in enumerate(bands):
        if lo != expected or hi < lo:
            raise ValueError("lead bands must be contiguous, ordered and cover 1..H")
        out[lo - 1 : hi] = b
        expected = hi + 1
    if expected != n_leads + 1:
        raise ValueError("lead bands must cover 1..H")
    return out


def band_label(band: tuple[int, int]) -> str:
    return f"h{band[0]:03d}-{band[1]:03d}"


# --------------------------------------------------------------------------- problems


def point_problem(
    origin_times: pd.DatetimeIndex,
    issue_every_steps: int,
    latency_steps: int,
    y: np.ndarray,
    point: np.ndarray,
    valid: np.ndarray,
    groups: Mapping[str, tuple[np.ndarray, int]],
    levels: Sequence[int],
    step_minutes: int = STEP_MINUTES,
) -> ConformalProblem:
    """Conformal intervals around a point forecast (method 1): signed residuals."""
    if np.any(valid & ~(np.isfinite(y) & np.isfinite(point))):
        raise ValueError("a valid pair has a missing actual or forecast")
    residual = np.where(valid, y.astype(np.float64) - point, np.nan)
    base = point.astype(np.float64)
    tails = {}
    for level in levels:
        tails[(level, "lo")] = Tail(0, base, negate=True)
        tails[(level, "hi")] = Tail(0, base)
    return ConformalProblem(
        origin_times,
        issue_every_steps,
        latency_steps,
        y.astype(np.float64),
        valid,
        [residual],
        tails,
        dict(groups),
        step_minutes,
    )


def quantile_problem(
    origin_times: pd.DatetimeIndex,
    issue_every_steps: int,
    latency_steps: int,
    y: np.ndarray,
    quantiles: Mapping[int, tuple[np.ndarray, np.ndarray]],
    valid: np.ndarray,
    groups: Mapping[str, tuple[np.ndarray, int]],
    step_minutes: int = STEP_MINUTES,
) -> ConformalProblem:
    """Conformalised quantile regression (method 3): one score matrix per tail.

    ``quantiles`` maps a level to its (lower, upper) quantile forecasts. The lower tail's
    score is ``q̂_lo − y`` and the upper tail's ``y − q̂_hi`` (Romano et al. 2019, the
    asymmetric version), so each side is corrected by its own quantile.
    """
    yy = y.astype(np.float64)
    scores: list[np.ndarray] = []
    tails: dict[tuple[int, str], Tail] = {}
    for level, (lo, hi) in quantiles.items():
        if np.any(valid & ~(np.isfinite(yy) & np.isfinite(lo) & np.isfinite(hi))):
            raise ValueError("a valid pair has a missing actual or quantile")
        for tail, base, score in (("lo", lo, lo - yy), ("hi", hi, yy - hi)):
            scores.append(np.where(valid, score, np.nan))
            tails[(level, tail)] = Tail(len(scores) - 1, base.astype(np.float64))
    return ConformalProblem(
        origin_times,
        issue_every_steps,
        latency_steps,
        yy,
        valid,
        scores,
        tails,
        dict(groups),
        step_minutes,
    )


def tod_labels(
    origin_times: pd.DatetimeIndex,
    n_leads: int,
    issue_every_steps: int,
    tz: str,
    block_starts: Sequence[int],
    step_minutes: int = STEP_MINUTES,
) -> tuple[np.ndarray, int]:
    """Time-of-day block of each pair's target, from the local hour of its start.

    ``block_starts`` are ascending local hours; hours before the first start belong to the
    last block (blocks may wrap past midnight). Leads are ``step_minutes`` long.
    """
    starts = np.asarray(block_starts)
    if starts.size == 0 or np.any(np.diff(starts) <= 0) or starts[0] < 0 or starts[-1] > 23:
        raise ValueError("block_starts must be ascending hours in 0..23")
    step = pd.Timedelta(minutes=step_minutes)
    n = len(origin_times)
    grid = pd.date_range(origin_times[0], periods=(n - 1) * issue_every_steps + n_leads, freq=step)
    hours = grid.tz_convert(tz).hour.to_numpy()
    block = np.searchsorted(starts, hours, side="right") - 1
    block[block < 0] = len(starts) - 1
    pos = np.arange(n)[:, None] * issue_every_steps + np.arange(n_leads)[None, :]
    return block[pos].astype(np.int8), len(starts)


# --------------------------------------------------------------------------- the run


def _scatter_add(target: np.ndarray, keys: np.ndarray, values: np.ndarray) -> None:
    """``target[keys[e]] += values[e]`` with repeated keys (few distinct keys per call)."""
    uniq, inv = np.unique(keys, return_inverse=True)
    onehot = (inv[None, :] == np.arange(len(uniq))[:, None]).astype(np.float64)
    sums = onehot @ values.reshape(len(keys), -1)
    target[uniq] += sums.reshape((len(uniq), *target.shape[1:]))


def _check_grid(times: pd.DatetimeIndex, issue_minutes: int) -> None:
    if len(times) < 2:
        raise ValueError("need at least two issue times")
    steps = np.diff(epoch_minutes(times))
    if np.any(steps != issue_minutes):
        raise ValueError("issue times must form a regular grid without gaps")


def run_conformal(
    problem: ConformalProblem,
    grid: CandidateGrid,
    levels: Sequence[int],
    retrains: Sequence[pd.Timestamp],
    bands: Sequence[tuple[int, int]],
    selection_weeks: float,
) -> ConformalResult:
    """Run every candidate stream and the per-retrain selection; return selected intervals.

    ``retrains`` are the selection instants (issue times on the grid); intervals are output
    from the first of them to the last issue time.
    """
    n, n_leads = problem.shape
    times = problem.origin_times
    step_minutes = problem.step_minutes
    issue_minutes = problem.issue_every_steps * step_minutes
    _check_grid(times, issue_minutes)
    lag = known_lag(np.arange(1, n_leads + 1), problem.issue_every_steps, problem.latency_steps)
    windows = [window_steps(w, issue_minutes) for w in grid.window_weeks]
    n_w, n_g, n_gam = grid.shape
    n_lv = len(levels)
    leads = np.arange(n_leads)
    band_of = band_index(bands, n_leads)
    n_b = len(bands)
    candidates = grid.candidates()
    for level in levels:
        for tail in TAILS:
            if (level, tail) not in problem.tails:
                raise ValueError(f"problem has no {tail} tail at {level}%")

    # Selection instants as issue-time indices.
    t_min = epoch_minutes(times)
    r_min = np.sort(epoch_minutes(retrains)) if len(retrains) else np.zeros(0, np.int64)
    r_idx = np.searchsorted(t_min, r_min)
    if r_idx.size == 0 or np.any(r_idx >= n) or np.any(t_min[np.minimum(r_idx, n - 1)] != r_min):
        raise ValueError("every retrain must be an issue time of the grid")
    retrain_at = {int(i): r for r, i in enumerate(r_idx)}
    span = round(selection_weeks * 7 * 1440)
    warm = t_min[0] + (max(windows) + int(lag.max())) * issue_minutes
    if r_min[0] - span < warm:
        raise ValueError(
            "the first selection slice starts before the longest window is full; "
            "start the selection later or shorten the windows"
        )
    i0 = int(r_idx[0])

    # One engine per (score matrix, grouping), holding every window length.
    engines: dict[tuple[int, int], RollingOrderStats] = {}
    for gi, gname in enumerate(grid.groupings):
        labels, n_groups = problem.groups[gname]
        for si, scores in enumerate(problem.scores):
            engines[(si, gi)] = RollingOrderStats(
                scores, problem.valid, labels, n_groups, lag, windows
            )
    queries: dict[int, list[tuple[int, int, Tail]]] = {si: [] for si in range(len(problem.scores))}
    for li, level in enumerate(levels):
        for ti, tail in enumerate(TAILS):
            spec = problem.tails[(level, tail)]
            queries[spec.score].append((li, ti, spec))

    # State: axes (H, W, G, Γ, Lv, tail) = leads, then candidates in grid order.
    alphas = np.array([alpha(level) for level in levels])
    beta_star = alphas / 2
    beta = np.broadcast_to(
        beta_star[None, None, None, None, :, None], (n_leads, n_w, n_g, n_gam, n_lv, 2)
    ).copy()
    gam = np.asarray(grid.gammas, dtype=np.float64)[None, None, None, :, None]
    ring_len = int(lag.max()) + 1
    ring = np.full((ring_len, n_leads, n_w, n_g, n_gam, n_lv, 2), np.nan)
    stream_shape = (n_w, n_g, n_gam, n_lv)
    n_streams = n_w * n_g * n_gam

    acc_sum = np.zeros((len(r_min) * n_b, *stream_shape))
    acc_cnt = np.zeros((len(r_min) * n_b, *stream_shape))
    day = (t_min + 600) // 1440  # NEM-time day number
    day0 = int(day[0])
    n_days = int(day[-1]) - day0 + 1
    diag = np.zeros((len(DIAGNOSTIC_STATS), n_days * n_b, *stream_shape))
    clipped = np.zeros((n_w, n_g, n_gam, n_lv, 2))
    produced = 0

    lower = {level: np.full((n - i0, n_leads), np.nan, np.float32) for level in levels}
    upper = {level: np.full((n - i0, n_leads), np.nan, np.float32) for level in levels}
    crossed = dict.fromkeys(levels, 0)
    chosen: np.ndarray | None = None  # (Lv, n_b) flat candidate index in grid order
    selections: list[Selection] = []
    bounds = np.empty((n_leads, n_w, n_g, n_gam, n_lv, 2))

    for i in range(n + ring_len - 1):
        if i < n:
            for eng in engines.values():
                eng.advance(i)

        # ---- outcomes that become known at issue time i: ACI, selection, diagnostics
        j = i - lag
        ok = (j >= 0) & (j < n)
        if ok.any():
            hh, jj = leads[ok], j[ok]
            past = ring[jj % ring_len, hh]  # (E, W, G, Γ, Lv, 2)
            lo, hi = past[..., 0], past[..., 1]
            yk = problem.y[jj, hh][:, None, None, None, None]
            has = (
                np.isfinite(lo) & np.isfinite(hi) & problem.valid[jj, hh][:, None, None, None, None]
            )
            below = has & (yk < lo)
            above = has & (yk > hi)
            if i < n:
                step = np.where(has, gam, 0.0)
                beta[hh, ..., 0] += step * (beta_star - below)
                beta[hh, ..., 1] += step * (beta_star - above)
                np.clip(beta, 0.0, BETA_MAX, out=beta)
            with np.errstate(invalid="ignore"):
                width = np.where(has, hi - lo, 0.0)
                score = np.where(
                    has,
                    width + (2 / alphas) * (np.maximum(lo - yk, 0.0) + np.maximum(yk - hi, 0.0)),
                    0.0,
                )
            target = t_min[jj] + step_minutes * (hh + 1)
            r = np.searchsorted(r_min, target, side="left")
            inc = r < len(r_min)
            inc[inc] &= t_min[jj[inc]] >= r_min[r[inc]] - span
            if inc.any():
                keys = r[inc] * n_b + band_of[hh[inc]]
                _scatter_add(acc_sum, keys, score[inc])
                _scatter_add(acc_cnt, keys, has[inc].astype(np.float64))
            keys = (day[jj] - day0) * n_b + band_of[hh]
            stats = (has, has & ~below & ~above, below, above, width, score)
            for s, values in enumerate(stats):
                _scatter_add(diag[s], keys, np.asarray(values, dtype=np.float64))

        if i >= n:
            continue

        # ---- selection at a retrain
        if i in retrain_at:
            r = retrain_at[i]
            chosen = np.empty((n_lv, n_b), dtype=np.int64)
            for b in range(n_b):
                cnt = acc_cnt[r * n_b + b].reshape(n_streams, n_lv)
                tot = acc_sum[r * n_b + b].reshape(n_streams, n_lv)
                for li, level in enumerate(levels):
                    c = cnt[:, li]
                    eligible = (c == c.max()) & (c > 0)
                    if not eligible.any():
                        raise ValueError(f"no pairs to select on at {times[i]}")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean = np.where(eligible, tot[:, li] / c, np.nan)
                    best = int(np.nanargmin(mean))
                    chosen[li, b] = best
                    selections.append(
                        Selection(
                            retrain=times[i].isoformat(),
                            level=level,
                            band=band_label(bands[b]),
                            chosen=candidates[best].key,
                            n_pairs=int(c.max()),
                            scores={cand.key: float(mean[s]) for s, cand in enumerate(candidates)},
                        )
                    )

        # ---- every candidate's intervals at issue time i
        for (si, gi), eng in engines.items():
            labels = problem.groups[grid.groupings[gi]][0][i]
            count = eng.counts(labels)[:, None, :]  # (W, 1, H)
            ks, clips = [], []
            for li, ti, spec in queries[si]:
                b_now = beta[:, :, gi, :, li, ti].transpose(1, 2, 0)  # (W, Γ, H)
                k, clip = conformal_rank(count, b_now)
                # A negated matrix holds minus the tail's score: its k-th smallest score
                # is minus the (n + 1 − k)-th smallest entry.
                ks.append(count + 1 - k if spec.negate else k)
                clips.append(clip)
            values = eng.kth(np.concatenate(ks, axis=1), labels)  # (W, Q·Γ, H)
            for q, (li, ti, spec) in enumerate(queries[si]):
                v = values[:, q * n_gam : (q + 1) * n_gam, :]
                if spec.negate:
                    v = -v
                sign = 1.0 if ti == 1 else -1.0
                bounds[:, :, gi, :, li, ti] = spec.base[i][:, None, None] + sign * v.transpose(
                    2, 0, 1
                )
                if i >= i0:
                    clipped[:, gi, :, li, ti] += clips[q].sum(axis=2)
        ring[i % ring_len] = bounds

        # ---- output the selected candidate per level and band
        if i >= i0:
            assert chosen is not None
            produced += n_leads
            for li, level in enumerate(levels):
                for b in range(n_b):
                    w, g, gm = np.unravel_index(chosen[li, b], (n_w, n_g, n_gam))
                    cols = band_of == b
                    lo_b = bounds[cols, w, g, gm, li, 0]
                    hi_b = bounds[cols, w, g, gm, li, 1]
                    cross = lo_b > hi_b
                    if cross.any():
                        crossed[level] += int(cross.sum())
                        mid = (lo_b + hi_b) / 2
                        lo_b = np.where(cross, mid, lo_b)
                        hi_b = np.where(cross, mid, hi_b)
                    lower[level][i - i0, cols] = lo_b
                    upper[level][i - i0, cols] = hi_b
        if i % 5000 == 0:
            log.debug("conformal.progress", issue=str(times[i]), done=f"{i / n:.0%}")

    return ConformalResult(
        origin_times=times[i0:],
        lower=lower,
        upper=upper,
        candidates=candidates,
        selections=selections,
        daily=_daily_frame(diag, day0, n_days, bands, candidates, levels),
        clipped=_clipped_frame(clipped, produced, candidates, levels),
        crossed=crossed,
    )


def _daily_frame(
    diag: np.ndarray,
    day0: int,
    n_days: int,
    bands: Sequence[tuple[int, int]],
    candidates: Sequence[Candidate],
    levels: Sequence[int],
) -> pd.DataFrame:
    n_b, n_c, n_lv = len(bands), len(candidates), len(levels)
    values = diag.reshape(len(DIAGNOSTIC_STATS), n_days, n_b, n_c, n_lv)
    d, b, c, lv = np.meshgrid(
        np.arange(n_days), np.arange(n_b), np.arange(n_c), np.arange(n_lv), indexing="ij"
    )
    dates = pd.to_datetime((day0 + d.ravel()) * 86400, unit="s").date
    frame = pd.DataFrame(
        {
            "nem_day": dates,
            "band": np.asarray([band_label(x) for x in bands])[b.ravel()],
            "candidate": np.asarray([x.key for x in candidates])[c.ravel()],
            "level": np.asarray(levels)[lv.ravel()],
        }
    )
    for s, name in enumerate(DIAGNOSTIC_STATS):
        frame[name] = values[s].ravel()
    frame["n"] = frame["n"].round().astype(np.int64)
    return frame.loc[frame["n"] > 0].reset_index(drop=True)


def _clipped_frame(
    clipped: np.ndarray, produced: int, candidates: Sequence[Candidate], levels: Sequence[int]
) -> pd.DataFrame:
    rows = []
    flat = clipped.reshape(len(candidates), len(levels), 2)
    for c, cand in enumerate(candidates):
        for li, level in enumerate(levels):
            for ti, tail in enumerate(TAILS):
                share = flat[c, li, ti] / produced if produced else float("nan")
                rows.append((cand.key, level, tail, share))
    return pd.DataFrame(rows, columns=["candidate", "level", "tail", "clipped_share"])
