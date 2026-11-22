"""Causality property tests for the anomaly detectors (m4_plan.md §2.8, criterion A5).

For random cut intervals ``c``, everything the detectors produce for intervals at or
before ``c`` (scores, p-values, window sizes, alert tiers and event openings) must be
bit-for-bit identical when

1. **truncation:** the data end at ``c`` (forecasts issued after ``c`` dropped);
2. **perturbation:** every input after ``c`` is randomised (market and supply values of
   intervals ending after ``c``, and forecasts issued after ``c``);

and 3. every month model of D6 (v1's Isolation Forest, v2's robust distance) was fitted only
on intervals ending by its month's start.

Controls show the tests have teeth: 4. a deliberately leaky profile (a window centred on the
day) fails tests 1 and 2, and perturbing data known at ``c`` does change the outputs. Every
test runs for both recipe versions (ids ``v1`` and ``v2``, m4_plan.md §2.10). The M4 report
runs this module for its version (A5).
"""

from functools import cache

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gridcast.anomalies import profiles
from gridcast.anomalies.detectors import DETECTORS, Recipe
from gridcast.anomalies.run import Detection, detect
from gridcast.anomalies.series import MARKET_ARRAYS, MarketSeries, QuantileBand
from tests.anomaly_fixtures import base_inputs, small_config, small_v2_recipe, split

CFG = small_config()
SPLIT = split()
RECIPES: dict[str, Recipe | None] = {"v1": None, "v2": small_v2_recipe()}
VERSIONS = pytest.mark.parametrize("version", list(RECIPES))


def _detect(series: MarketSeries, band: QuantileBand, version: str) -> Detection:
    return detect(series, band, SPLIT, CFG, recipe=RECIPES[version])


@cache
def _baseline(version: str) -> Detection:
    return _detect(*base_inputs(), version)


def _cut(share: float) -> int:
    """A cut inside the scored period, not at the very end."""
    series, _ = base_inputs()
    first = int(np.flatnonzero(SPLIT.masks(series.ends)[DETECTORS[0]])[0])
    return first + int(share * (len(series) - 2 - first))


def _differences(a: Detection, b: Detection, c: int) -> list[str]:
    """What differs at or before interval ``c``."""
    out = []
    upto = slice(0, c + 1)
    for d in a.run.recipe.detectors:
        for label, x, y in (
            ("score", a.run.scores[d], b.run.scores[d]),
            ("p", a.run.pvalues[d], b.run.pvalues[d]),
            ("n_window", a.run.n_window[d], b.run.n_window[d]),
        ):
            if not np.array_equal(x[upto], y[upto], equal_nan=True):
                out.append(f"{d} {label}")
    for d, t in a.tiers.items():
        if not np.array_equal(t[upto], b.tiers[d][upto]):
            out.append(f"{d} tier")
    opened_a = [e.first for e in a.events if e.first <= c]
    opened_b = [e.first for e in b.events if e.first <= c]
    if opened_a != opened_b:
        out.append("event openings")
    return out


def _truncated(c: int, version: str) -> Detection:
    series, band = base_inputs()
    return _detect(series.head(c + 1), band.issued_until(series.ends[c]), version)


def _randomised_after(c: int, seed: int, version: str) -> Detection:
    series, band = base_inputs()
    rng = np.random.default_rng(seed)
    after = np.arange(len(series)) > c
    arrays = {}
    for name in MARKET_ARRAYS:
        values = getattr(series, name).copy()
        scale = np.nanstd(values) + 1.0
        values[after] = rng.normal(np.nanmean(values), 3 * scale, after.sum())
        arrays[name] = values
    later = np.asarray(band.issue_times > series.ends[c])
    lower, upper, valid = band.lower.copy(), band.upper.copy(), band.valid.copy()
    lower[later] += rng.normal(0, 800, lower[later].shape)
    upper[later] = lower[later] + rng.uniform(0, 3000, lower[later].shape)
    valid[later] = rng.random(valid[later].shape) < 0.5
    valid &= np.isfinite(lower) & np.isfinite(upper)
    return _detect(
        series.with_values(**arrays), QuantileBand(band.issue_times, lower, upper, valid), version
    )


@VERSIONS
@settings(max_examples=8, deadline=None)
@given(share=st.floats(0.0, 1.0))
def test_truncation_leaves_the_past_unchanged(version: str, share: float) -> None:
    c = _cut(share)
    assert _differences(_baseline(version), _truncated(c, version), c) == []


@VERSIONS
@settings(max_examples=8, deadline=None)
@given(share=st.floats(0.0, 1.0), seed=st.integers(0, 2**31))
def test_randomising_the_future_leaves_the_past_unchanged(
    version: str, share: float, seed: int
) -> None:
    c = _cut(share)
    assert _differences(_baseline(version), _randomised_after(c, seed, version), c) == []


@VERSIONS
def test_month_models_are_fitted_before_their_month(version: str) -> None:
    det = _baseline(version)
    fits = det.run.state.month_fits()
    assert fits
    for mm in fits:
        assert mm.train_last_end <= mm.start


@VERSIONS
def test_the_comparison_sees_every_output(version: str) -> None:
    """The baseline has alerts and events before the cut, so the tests compare something."""
    det = _baseline(version)
    c = _cut(0.5)
    assert any(e.first <= c for e in det.events)
    assert det.run.recipe.detectors[:5] == DETECTORS[:5]
    for d in det.run.recipe.detectors:
        assert np.isfinite(det.run.pvalues[d][: c + 1]).any(), d


# ------------------------------------------------------------------------------ controls


def _centred(day: int, days: int) -> np.ndarray:
    """A leaky profile window: centred on the day, so it uses later days."""
    return np.arange(day - days // 2, day + days - days // 2)


@VERSIONS
def test_control_leaky_profile_fails_truncation(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _cut(0.5)
    monkeypatch.setattr(profiles, "lookback_days", _centred)
    leaky_full = _detect(*base_inputs(), version)
    assert _differences(leaky_full, _truncated(c, version), c) != []


@VERSIONS
def test_control_leaky_profile_fails_perturbation(
    version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = _cut(0.5)
    monkeypatch.setattr(profiles, "lookback_days", _centred)
    leaky_full = _detect(*base_inputs(), version)
    assert _differences(leaky_full, _randomised_after(c, 0, version), c) != []


@VERSIONS
def test_control_perturbing_known_data_changes_the_outputs(version: str) -> None:
    series, band = base_inputs()
    c = _cut(0.5)
    demand = series.demand.copy()
    demand[c - 288 : c + 1] += 3000  # the last day before the cut, known at c
    det = _detect(series.with_values(demand=demand), band, version)
    assert _differences(_baseline(version), det, c) != []
