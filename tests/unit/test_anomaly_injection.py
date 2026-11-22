"""The injection harness (m4_plan.md §2.9, A3) on synthetic data."""

from functools import cache

import numpy as np
import pytest

from gridcast.anomalies.detectors import (
    DEMAND_LEVEL,
    ISOLATION_FOREST,
    PRICE_LEVEL,
    PRICE_VOLATILITY,
    RESIDUAL_DAYAHEAD,
    RESIDUAL_SHORT,
    compute_scores,
)
from gridcast.anomalies.injection import (
    FLAT_LABEL,
    PRICE_ALTERNATE,
    PRICE_FLAT,
    PRICE_SET,
    draw_starts,
    eligible_starts,
    injected_scores,
    injection_specs,
    power_table,
    run_injections,
)
from gridcast.anomalies.run import DetectorRun, run_detectors
from gridcast.anomalies.series import epoch_minutes
from tests.anomaly_fixtures import base_inputs, small_config, split

CFG = small_config()
SPLIT = split()
SPECS = {s.label: s for s in injection_specs(CFG)}


@cache
def _run() -> DetectorRun:
    series, band = base_inputs()
    return run_detectors(series, band, CFG)


@cache
def _results() -> object:
    series, _ = base_inputs()
    return run_injections(series, _run(), SPLIT, CFG, n_starts=150)


SPACING = 5 * 288  # longer than the 3-day profiles and the 1-day D5 baseline


def _starts(label: str) -> np.ndarray:
    """Eligible starts whose span stays inside one local day, five days apart.

    A full recompute then sees each injection exactly as the harness does: the injected
    day's profile is from earlier days, and no injection reaches another's profiles,
    baselines or windows.
    """
    spec = SPECS[label]
    run = _run()
    masks = SPLIT.masks(run.ends)
    elig = eligible_starts(spec, run, masks[spec.detector], CFG.pvalues.notable, 12)
    day = run.state.lay.local_day
    elig = draw_starts(elig[day[elig] == day[elig + spec.span - 1]], 400, 7, 1)
    picked: list[int] = []
    for i in elig:
        if not picked or i - picked[-1] > SPACING + spec.span:
            picked.append(int(i))
    return np.asarray(picked)


def _modified(label: str, size: float, starts: np.ndarray):
    series, band = base_inputs()
    spec = SPECS[label]
    demand, price = series.demand.copy(), series.price.copy()
    for i in starts:
        rows = np.arange(i, i + spec.span)
        if spec.kind == PRICE_SET:
            price[rows] = size
        elif spec.kind == PRICE_ALTERNATE:
            price[rows] += size * np.where(np.arange(spec.span) % 2 == 0, 1.0, -1.0)
        elif spec.kind == PRICE_FLAT:
            price[rows[1:]] = price[i]
        else:
            demand[rows] += size
    return series.with_values(demand=demand, price=price), band


@pytest.mark.parametrize(
    ("label", "size"),
    [
        (DEMAND_LEVEL, 700.0),
        (DEMAND_LEVEL, -1500.0),
        (RESIDUAL_SHORT, 400.0),
        (RESIDUAL_DAYAHEAD, -900.0),
        (PRICE_LEVEL, 15000.0),
        (PRICE_LEVEL, -300.0),
        (PRICE_VOLATILITY, 200.0),
        (FLAT_LABEL, 0.0),
        (ISOLATION_FOREST, 1000.0),
    ],
)
def test_injected_scores_match_a_full_recompute(label: str, size: float) -> None:
    """Where the unmodified context is all the span sees, the harness equals a recompute.

    Two parts of the context are kept unmodified by design (§2.9), so they are compared
    only where a recompute leaves them alone:

    * the D5 baseline lags 1 h: a 3-hour flat injection enters it after 13 intervals;
    * the Isolation Forest models: a recompute would refit a later month on an earlier
      injection, so D6 compares the recomputed features under the unmodified models.
    """
    starts = _starts(label)
    assert len(starts) >= 3
    spec = SPECS[label]
    series, _ = base_inputs()
    run = _run()
    harness = injected_scores(spec, size, starts, series, run, CFG)
    scores, state = compute_scores(*_modified(label, size, starts), CFG)
    rows = starts[:, None] + np.arange(spec.span)[None, :]
    if spec.detector == ISOLATION_FOREST:
        months = run.state.lay.nem_month[rows]
        expected = np.full(rows.shape, np.nan)
        for m in np.unique(months):
            sel = months == m
            expected[sel] = run.state.models[int(m)].score(state.features[rows][sel])
    else:
        expected = scores[spec.detector][rows]
    cols = slice(0, 13) if spec.kind == PRICE_FLAT else slice(None)
    np.testing.assert_allclose(harness[:, cols], expected[:, cols], rtol=1e-12, atol=1e-12)


def test_eligible_starts_respect_period_validity_guard_and_alignment() -> None:
    run = _run()
    masks = SPLIT.masks(run.ends)
    notable = CFG.pvalues.notable
    for spec in SPECS.values():
        elig = eligible_starts(spec, run, masks[spec.detector], notable, 12)
        assert elig.size, spec.label
        p = run.pvalues[spec.detector]
        for i in elig[:: max(1, len(elig) // 50)]:
            span = np.arange(i, i + spec.span)
            assert masks[spec.detector][span].all()
            assert np.isfinite(run.scores[spec.detector][span]).all()
            guard = np.arange(max(0, i - 12), min(len(p), i + spec.span + 12))
            assert not (p[guard] <= notable).any()
        if spec.aligned:
            assert ((epoch_minutes(run.ends[elig]) - 5) % 30 == 0).all()


def test_starts_are_shared_by_sizes_and_signs_and_seeded() -> None:
    res = _results()
    for label, grp in res.groupby("label"):
        starts = {size: tuple(g["start"]) for size, g in grp.groupby("size")}
        assert len(set(starts.values())) == 1, label
    again = run_injections(base_inputs()[0], _run(), SPLIT, CFG, n_starts=150)
    assert again.equals(res)


def test_largest_sizes_are_detected_more_often() -> None:
    table = power_table(_results())
    for label in (DEMAND_LEVEL, PRICE_LEVEL, RESIDUAL_SHORT):
        t = table[table["label"] == label].set_index("size")["power"]
        pos = t[t.index > 0].sort_index()
        assert pos.iloc[-1] >= pos.iloc[0], label
        assert pos.iloc[-1] > 0.5, label


def test_result_columns_and_flat_injection_is_reported_only() -> None:
    res = _results()
    assert set(res["label"]) == set(SPECS)
    assert res.loc[res["label"] == FLAT_LABEL, "reported_only"].all()
    assert not res.loc[res["label"] != FLAT_LABEL, "reported_only"].any()
    assert (res["min_p"] > 0).all() and (res["min_p"] <= 1).all()
    assert (res["detected"] == (res["min_p"] <= CFG.pvalues.notable)).all()
