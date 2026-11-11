"""Scoring stored interval batches for the M3 development report (m3_plan.md §2.4, §3, step 4).

Pure functions on (issue time × lead) matrices. :mod:`gridcast.intervals.report` loads the
stored batches and writes the report.

* :func:`pair_stats` gives every per-pair statistic of one method's bounds: coverage, share
  below and above, width and interval score per level; the share of actuals at or below each
  bound and its pinball loss (each bound is a quantile τ = ½ ∓ L/200); the WIS; and nesting
  failures between neighbouring levels.
* :class:`Aggregator` sums a statistic over the scored pairs by NEM issue day and lead, and
  by issue day and a pair label (target hour, target month) or a pair group (extreme events).
  Day sums are what the block bootstrap resamples.
* :func:`bootstrap_ratio` is M2's day-block bootstrap of a pooled mean (or of a difference
  of means on the same pairs), with optional longer blocks for the sensitivity check.
* :func:`success_criteria` computes U1–U4 exactly as pre-registered.
* :func:`rearrange_bounds` sorts the bounds of every pair across levels (the reported-only
  *CQR, rearranged* row, user decision 2026-11-12).
* :func:`forecast_day_max` builds the forecast daily maximum temperature known at each issue
  time (the hot and cold groups known at issue time).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from gridcast.evaluation.m2_report import Criterion, LeakageOutcome
from gridcast.evaluation.report import horizon_label
from gridcast.intervals.quantile import quantile_key
from gridcast.intervals.scores import alpha, interval_score, pinball, weighted_interval_score

BOOTSTRAP_DRAWS = 2000  # as M2's AEMO comparison (evaluation.aemo_benchmark)
U1_RANGE = (78.0, 82.0)  # pooled 80% coverage, % (m3_plan.md §3)
U2_RANGE = (75.0, 85.0)  # 80% coverage at every lead, %
U3_FIRST_STEP = 24  # U3 per-lead part: every lead from 2 h (h = 24) to 24 h


def tau_of(level: int, side: str) -> float:
    """Quantile level of a central interval's bound: 80% → 0.1 (lower) and 0.9 (upper)."""
    half = level / 200.0
    return round(0.5 - half if side == "lo" else 0.5 + half, 6)


# --------------------------------------------------------------------------- per pair


def rearrange_bounds(
    lower: Mapping[int, np.ndarray], upper: Mapping[int, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Sort each pair's bounds across levels, read as quantiles in ascending τ.

    The 2K bounds of K levels are, by τ, lo(widest) … lo(narrowest), hi(narrowest) …
    hi(widest). Sorting them per pair makes every level nested inside the next wider one
    (rearrangement, Chernozhukov et al. 2010, as step 2 did for raw QR). A pair with any
    missing bound gets NaN at every level.
    """
    levels = sorted(lower)
    order = [lower[lv] for lv in reversed(levels)] + [upper[lv] for lv in levels]
    stacked = np.stack(order)
    missing = ~np.isfinite(stacked).all(axis=0)
    stacked = np.sort(stacked, axis=0)
    stacked[:, missing] = np.nan
    k = len(levels)
    new_lower = {lv: stacked[k - 1 - i] for i, lv in enumerate(levels)}
    new_upper = {lv: stacked[k + i] for i, lv in enumerate(levels)}
    return new_lower, new_upper


def nesting_failures(
    lower: Mapping[int, np.ndarray], upper: Mapping[int, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per neighbouring pair of levels (narrow, wider): where the wider level's lower bound
    is above the narrower one's (``lo_N_W``), or its upper bound below (``hi_N_W``)."""
    out = {}
    for narrow, wider in pairwise(sorted(lower)):
        out[f"lo_{narrow}_{wider}"] = lower[wider] > lower[narrow]
        out[f"hi_{narrow}_{wider}"] = upper[wider] < upper[narrow]
    return out


def pair_stats(
    lower: Mapping[int, np.ndarray],
    upper: Mapping[int, np.ndarray],
    actual: np.ndarray,
    point: np.ndarray,
) -> Iterator[tuple[str, np.ndarray]]:
    """Every per-pair statistic of one method's bounds (float64 matrices), one at a time.

    Names: ``cov_L``, ``below_L``, ``above_L``, ``width_L``, ``is_L`` per level ``L``;
    ``le_qXXX`` (actual at or below the bound) and ``pin_qXXX`` (pinball loss) per bound;
    ``wis`` (``point`` as the median); ``non_nested`` and one ``nest_*`` per side and pair
    of neighbouring levels (:func:`nesting_failures`). Values outside the scored pairs are
    meaningless (NaN or arbitrary); the aggregator masks them.
    """
    y = np.asarray(actual, dtype=np.float64)
    bounds = {lv: (lower[lv].astype(np.float64), upper[lv].astype(np.float64)) for lv in lower}
    with np.errstate(invalid="ignore"):
        for level, (lo, hi) in bounds.items():
            yield f"cov_{level}", ((lo <= y) & (y <= hi)).astype(np.float64)
            yield f"below_{level}", (y < lo).astype(np.float64)
            yield f"above_{level}", (y > hi).astype(np.float64)
            yield f"width_{level}", hi - lo
            yield f"is_{level}", interval_score(lo, hi, y, alpha(level))
        for level, (lo, hi) in bounds.items():
            for side, bound in (("lo", lo), ("hi", hi)):
                tau = tau_of(level, side)
                key = quantile_key(tau)
                yield f"le_{key}", (y <= bound).astype(np.float64)
                yield f"pin_{key}", pinball(bound, y, tau)
        yield "wis", weighted_interval_score(point, bounds, y)
        nesting = nesting_failures(lower, upper)
        any_fail = np.zeros(y.shape, dtype=bool)
        for name, fail in nesting.items():
            any_fail |= fail
            yield f"nest_{name}", fail.astype(np.float64)
        yield "non_nested", any_fail.astype(np.float64)


# --------------------------------------------------------------------------- aggregation


def issue_day_codes(day_ordinal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Codes 0…D−1 of each issue row's day (rows sorted by time), and each code's ordinal."""
    if np.any(np.diff(day_ordinal) < 0):
        raise ValueError("issue rows must be sorted by time")
    days, codes = np.unique(day_ordinal, return_inverse=True)
    return codes.astype(np.int64), days.astype(np.int64)


@dataclass(slots=True)
class StatSums:
    """Sums of one statistic over the scored pairs."""

    by_lead: np.ndarray  # (D, H): per issue day and lead
    by_label: dict[str, np.ndarray] = field(default_factory=dict)  # (D, n_labels)
    by_group: dict[str, np.ndarray] = field(default_factory=dict)  # (D,)

    @property
    def by_day(self) -> np.ndarray:
        return self.by_lead.sum(axis=1)


class Aggregator:
    """Sums statistics over ``mask`` by issue day and lead, by labels and by groups.

    ``day_code`` (n,) gives each issue row's day code (:func:`issue_day_codes`);
    ``labels`` maps a name to a per-pair label matrix with values in ``[0, n_labels)``;
    ``groups`` maps a name to a per-pair boolean matrix (groups may overlap).
    """

    def __init__(
        self,
        mask: np.ndarray,
        day_code: np.ndarray,
        labels: Mapping[str, tuple[np.ndarray, int]] | None = None,
        groups: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        if mask.ndim != 2 or day_code.shape != (mask.shape[0],):
            raise ValueError("mask is (issue times × leads) and day_code has one code per row")
        self.mask = mask
        self.n_days = int(day_code.max()) + 1
        starts = np.flatnonzero(np.r_[True, np.diff(day_code) != 0])
        if len(starts) != self.n_days or np.any(day_code[starts] != np.arange(self.n_days)):
            raise ValueError("day codes must run 0…D−1 over contiguous rows")
        self._starts = starts
        flat = np.flatnonzero(mask.ravel())
        self._flat = flat
        row_day = np.repeat(day_code, mask.shape[1])[flat]
        self._labels: dict[str, tuple[np.ndarray, int]] = {}
        for name, (lab, n_labels) in (labels or {}).items():
            values = lab.ravel()[flat].astype(np.int64)
            if values.size and (values.min() < 0 or values.max() >= n_labels):
                raise ValueError(f"label {name!r} outside [0, {n_labels})")
            self._labels[name] = (row_day * n_labels + values, n_labels)
        self._groups: dict[str, np.ndarray] = {}
        for name, member in (groups or {}).items():
            inside = member.ravel()[flat]
            self._groups[name] = np.flatnonzero(inside)
        self._group_day = row_day
        self.counts = self.add(np.ones(mask.shape))

    def add(self, values: np.ndarray, conditional: bool = True) -> StatSums:
        """Sums of ``values`` over the scored pairs (labels and groups only if ``conditional``)."""
        masked = np.where(self.mask, values, 0.0)
        by_lead = np.add.reduceat(masked, self._starts, axis=0)
        out = StatSums(by_lead)
        if not conditional:
            return out
        v = values.ravel()[self._flat]
        for name, (keys, n_labels) in self._labels.items():
            sums = np.bincount(keys, weights=v, minlength=self.n_days * n_labels)
            out.by_label[name] = sums.reshape(self.n_days, n_labels)
        for name, idx in self._groups.items():
            out.by_group[name] = np.bincount(
                self._group_day[idx], weights=v[idx], minlength=self.n_days
            )
        return out


# --------------------------------------------------------------------------- bootstrap


@dataclass(frozen=True, slots=True)
class Estimate:
    """A pooled mean (or difference of means) with a block-bootstrap 95% interval."""

    value: float
    lo: float
    hi: float
    n_pairs: int
    n_blocks: int


def bootstrap_ratio(
    sums: np.ndarray,
    counts: np.ndarray,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
    blocks: np.ndarray | None = None,
) -> Estimate:
    """Pooled ``Σ sums / Σ counts`` with a 95% percentile interval from resampling blocks.

    ``sums`` and ``counts`` are per issue day. Without ``blocks`` each day is a block (M2's
    day-block bootstrap); ``blocks`` maps each day to a longer block (the 7-day sensitivity).
    Blocks without pairs are dropped. For a difference of two methods on the same pairs,
    pass the difference of their day sums.
    """
    s = np.asarray(sums, dtype=np.float64)
    n = np.asarray(counts, dtype=np.float64)
    if blocks is not None:
        n_blocks = int(blocks.max()) + 1
        s = np.bincount(blocks, weights=s, minlength=n_blocks)
        n = np.bincount(blocks, weights=n, minlength=n_blocks)
    keep = n > 0
    s, n = s[keep], n[keep]
    total = float(n.sum())
    if total == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(draws, len(s)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return Estimate(float(s.sum() / total), float(lo), float(hi), int(total), len(s))


def week_blocks(day_ordinals: np.ndarray, block_days: int = 7) -> np.ndarray:
    """Block of each issue day: consecutive ``block_days``-day weeks from the first day."""
    return ((day_ordinals - day_ordinals[0]) // block_days).astype(np.int64)


# --------------------------------------------------------------------------- criteria


def per_lead(sums: StatSums, counts: StatSums) -> np.ndarray:
    """Mean per lead (H,)."""
    n = counts.by_lead.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(sums.by_lead.sum(axis=0) / n)


def pooled(sums: StatSums, counts: StatSums) -> float:
    return float(sums.by_lead.sum() / counts.by_lead.sum())


def success_criteria(
    cov80_lead: np.ndarray,
    cov80_pooled: float,
    is80_lead: np.ndarray,
    is80_pooled: float,
    baseline_is80_lead: Mapping[str, np.ndarray],
    leakage: LeakageOutcome,
    label: Mapping[str, str] | None = None,
) -> list[Criterion]:
    """U1–U4 of m3_plan.md §3 for the headline (CQR on LightGBM C at 80%).

    Coverage in %, interval scores in MW; per-lead arrays have one value per lead h = 1…H.
    U3's pooled part compares with the mean over every lead of the per-lead baseline minimum,
    as M2's S2 did.
    """
    label = label or {}
    leads = np.arange(1, len(cov80_lead) + 1)
    lo1, hi1 = U1_RANGE
    lo2, hi2 = U2_RANGE
    inside = (cov80_lead >= lo2) & (cov80_lead <= hi2)
    worst_low, worst_high = int(np.nanargmin(cov80_lead)), int(np.nanargmax(cov80_lead))

    names = list(baseline_is80_lead)
    stacked = np.stack([baseline_is80_lead[m] for m in names])
    envelope = stacked.min(axis=0)
    best = [names[i] for i in stacked.argmin(axis=0)]
    window = leads >= U3_FIRST_STEP
    margin = (is80_lead - envelope)[window]
    closest = int(np.argmax(margin))
    closest_lead = int(leads[window][closest])
    closest_best = label.get(best[closest_lead - 1], best[closest_lead - 1])
    mean_envelope = float(envelope.mean())
    u3_leads = bool(np.all(margin < 0))
    u3_pooled = bool(is80_pooled < mean_envelope)

    tests = (
        f"{leakage.passed} passed, {leakage.failed} failed"
        if leakage.returncode >= 0
        else "not run"
    )
    return [
        Criterion(
            "U1",
            f"pooled 80% coverage ∈ [{lo1:g}%, {hi1:g}%]",
            f"{cov80_pooled:.2f}%",
            bool(lo1 <= cov80_pooled <= hi1),
        ),
        Criterion(
            "U2",
            f"80% coverage(h) ∈ [{lo2:g}%, {hi2:g}%] for every h from 5 min to 24 h",
            f"{int(inside.sum())} of {len(leads)} lead times; lowest {cov80_lead[worst_low]:.2f}% "
            f"at {horizon_label(int(leads[worst_low]))}, highest "
            f"{cov80_lead[worst_high]:.2f}% at {horizon_label(int(leads[worst_high]))}",
            bool(inside.all()),
        ),
        Criterion(
            "U3",
            "IS₈₀(h) < min over baseline intervals of IS₈₀(h) for every h from 2 h to 24 h, "
            "and pooled IS₈₀ < mean over h of that minimum",
            f"{int((margin < 0).sum())} of {len(margin)} lead times; closest at "
            f"{horizon_label(closest_lead)}: {is80_lead[closest_lead - 1]:,.1f} vs "
            f"{envelope[closest_lead - 1]:,.1f} MW ({closest_best}); "
            f"pooled {is80_pooled:,.1f} vs {mean_envelope:,.1f} MW",
            u3_leads and u3_pooled,
        ),
        Criterion(
            "U4",
            "every interval leakage property test passes (run by this report)",
            f"leakage tests {tests}",
            leakage.ok,
        ),
    ]


# --------------------------------------------------------------------------- weather groups


def forecast_day_max(
    origin_ns: np.ndarray,
    target_pos: np.ndarray,
    local_day: np.ndarray,
    forecasts: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Forecast daily maximum of each pair's target local day, as known at its issue time.

    ``local_day`` (grid,) is sorted and holds every interval of each target day;
    ``target_pos`` (n, H) indexes that grid; ``forecasts`` lists ``(values, available_ns)``
    on the same grid, freshest first. For every interval of the target's day the freshest
    value published by the issue time is used (the rule of C's weather features,
    :func:`gridcast.features.weather.select_forecast`), and the day's maximum is taken over
    the intervals that have one. NaN if none has.
    """
    if np.any(np.diff(local_day) < 0):
        raise ValueError("local_day must be sorted")
    tday = local_day[target_pos]
    first = tday[:, 0]
    offset = tday - first[:, None]  # local days after the first lead's (≤ 2, 23 h DST days)
    n_offsets = int(offset.max()) + 1
    out = np.full((len(first), n_offsets), np.nan)
    for d in np.unique(tday):
        lo, hi = np.searchsorted(local_day, d, "left"), np.searchsorted(local_day, d, "right")
        if hi == lo:
            raise ValueError(f"local day {d} is not on the weather grid")
        k = d - first
        rows = np.flatnonzero((k >= 0) & (k < n_offsets) & (tday[:, -1] >= d))
        o = origin_ns[rows][:, None]
        val = np.full((len(rows), hi - lo), np.nan)
        for values, available in forecasts:
            v, a = values[lo:hi][None, :], available[lo:hi][None, :]
            take = np.isnan(val) & (a <= o) & np.isfinite(v)
            val = np.where(take, v, val)
        with warnings.catch_warnings():  # a day without any known value is NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            out[rows, k[rows]] = np.nanmax(val, axis=1)
    return np.take_along_axis(out, offset, axis=1)
