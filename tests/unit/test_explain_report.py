"""M4 step 5's report maths on synthetic data: price-association bins and rates, the day-block
bootstrap, and the SHAP summaries of a stored sample (m4_plan.md §1 "Clarified in step 5")."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast.config import load_config
from gridcast.explainability.associations import (
    AssociationData,
    Binning,
    Bootstrap,
    bin_rates,
    compute_associations,
    development_binnings,
    fixed_codes,
    nem_days,
    quantile_edges,
)
from gridcast.explainability.families import families
from gridcast.explainability.global_run import (
    FITS_DIR,
    MANIFEST,
    StoredSample,
    shap_col,
    x_col,
)
from gridcast.explainability.report import BannedPhraseError, _disclosures, _how_to_read
from gridcast.explainability.summary import band_label, summarise

CFG = load_config()
AC = CFG.explain_report.associations
SC = CFG.explain_report.shap


# --------------------------------------------------------------------------- bins


def test_deciles_are_right_closed_and_missing_values_get_minus_one() -> None:
    v = np.arange(1, 101, dtype=float)
    edges = quantile_edges(v, 10)
    assert len(edges) == 9
    b = Binning("demand", "quantile", edges)
    codes = b.codes(np.array([edges[0], edges[0] + 1e-9, np.nan, 1e9, -1e9]))
    assert codes.tolist() == [0, 1, -1, 9, 0]
    assert np.bincount(b.codes(v), minlength=10).tolist() == [10] * 10
    labels = b.labels("MW")
    assert len(labels) == 10 and labels[0].startswith("≤") and labels[-1].startswith(">")


def test_solar_has_a_zero_bin_plus_deciles_of_positive_output() -> None:
    v = np.r_[np.zeros(500), np.arange(1, 501, dtype=float)]
    b = Binning("solar", "zero_quantile", quantile_edges(v[v > 0], 10))
    codes = b.codes(v)
    assert b.n_bins == 11 and (codes[:500] == 0).all()
    assert np.bincount(codes[500:], minlength=11).tolist() == [0] + [50] * 10
    assert b.labels("MW")[0].startswith("0 MW (zero")
    with pytest.raises(ValueError, match="negative"):
        b.codes(np.array([-1.0]))


def test_fixed_bins_include_their_lower_edge() -> None:
    codes, present = fixed_codes(np.array([4.99, 5.0, 9.99, 10.0, np.nan, -0.1]), 5.0)
    assert present.tolist() == [-1, 0, 1, 2]
    assert codes.tolist() == [1, 2, 2, 3, -1, 0]


def test_categories() -> None:
    b = Binning("hour", "category", categories=tuple(range(24)))
    assert b.codes(np.array([0, 23, 12])).tolist() == [0, 23, 12] and b.n_bins == 24


def test_nem_days_go_by_interval_start() -> None:
    ends = pd.DatetimeIndex(["2024-01-01 14:00", "2024-01-01 14:05"], tz="UTC")  # 00:00, 00:05
    days = nem_days(ends)
    assert days[1] - days[0] == 1  # the first interval started on the previous NEM day


# --------------------------------------------------------------------------- rates


def _data(n_days: int = 40, seed: int = 0) -> AssociationData:
    rng = np.random.default_rng(seed)
    n = 288 * n_days
    ends = pd.date_range("2024-01-01 14:05", periods=n, freq="5min", tz="UTC")
    demand = rng.normal(5000, 800, n)
    price = np.where(demand > 6000, 400.0, np.where(demand < 4200, -10.0, 80.0))
    solar = np.where(rng.random(n) < 0.45, 0.0, rng.gamma(2.0, 300.0, n))
    obs = {
        "demand": demand,
        "temperature": rng.normal(18, 6, n),
        "wind": rng.gamma(3.0, 400.0, n),
        "solar": solar,
        "net_interchange": rng.normal(0, 500, n),
        "hour": np.arange(n) // 12 % 24,
        "month": np.ones(n, dtype=np.int64),
    }
    obs["wind"][:7] = np.nan
    return AssociationData(ends, nem_days(ends), price, obs)


def test_rates_and_bootstrap_intervals() -> None:
    data = _data()
    boot = Bootstrap.draw(data.day, 200, (1, 2), 0.95)
    assert boot.weights.shape == (200, 40) and (boot.weights.sum(axis=1) == 40).all()
    edges = quantile_edges(data.observed["demand"], 10)
    codes = Binning("demand", "quantile", edges).codes(data.observed["demand"])
    t = bin_rates(data, codes, 10, boot, AC)
    assert t["n"].sum() == len(data.day) and np.isclose(t["share"].sum(), 1)
    for b in range(10):  # rates equal direct counts
        sel = codes == b
        assert t["high"][b] == pytest.approx((data.price[sel] > 300).mean())
        assert t["low"][b] == pytest.approx((data.price[sel] < 0).mean())
    assert t["high"].iloc[-1] > 0.5 and t["high"].iloc[0] == 0  # the planted association
    ok = t["high_lo"] <= t["high"] + 1e-12
    assert ok.all() and (t["high"] <= t["high_hi"] + 1e-12).all()
    same = Bootstrap.draw(data.day, 200, (1, 2), 0.95)
    np.testing.assert_array_equal(same.weights, boot.weights)  # seeded


def test_a_variable_with_missing_values_drops_only_those_intervals() -> None:
    data = _data()
    res = compute_associations(data, AC, 7, dt.date(2024, 1, 2), dt.date(2024, 2, 10))
    assert res.tables["wind"].missing == 7 and res.tables["demand"].missing == 0
    assert res.tables["wind"].frame["n"].sum() == len(data.day) - 7
    assert len(res.two_way) == 9 and res.two_way["n"].sum() == len(data.day) - 7
    assert res.overall["high"].iloc[0] == pytest.approx((data.price > 300).mean())


def test_holdout_reuses_development_edges() -> None:
    dev, other = _data(seed=0), _data(seed=1)
    bins = development_binnings(dev, AC)
    res = compute_associations(other, AC, 7, dt.date(2024, 1, 2), dt.date(2024, 2, 10), bins)
    assert res.binnings["demand"].edges == bins["demand"].edges
    assert res.binnings["demand"].edges != development_binnings(other, AC)["demand"].edges


# --------------------------------------------------------------------------- SHAP summaries


def _sample(tmp_path: Path) -> StoredSample:
    """Two retrains × two bands; the second band switches target form between retrains."""
    rng = np.random.default_rng(3)
    columns = list(families("C").columns)
    fits = []
    (tmp_path / FITS_DIR).mkdir()
    specs = [
        ("2024-01-01", (1, 6), "latest"),
        ("2024-04-01", (1, 6), "latest"),
        ("2024-01-01", (73, 288), "direct"),
        ("2024-04-01", (73, 288), "residual"),
    ]
    for retrain, band, form in specs:
        n = 400
        key = f"{retrain}_{band[0]}"
        X = rng.normal(1000, 200, size=(n, len(columns))).astype(np.float32)
        shap = rng.normal(0, 50, size=(n, len(columns)))
        base = np.full(n, 30.0)
        anchor = {
            "direct": np.zeros(n),
            "latest": X[:, columns.index("dem_now")].astype(np.float64),
            "residual": X[:, columns.index("dem_same_time_yesterday")].astype(np.float64),
        }[form]
        raw = base + shap.sum(axis=1)
        issue = pd.date_range(f"{retrain} 00:00", periods=n, freq="22h", tz="UTC")  # a year
        lead = np.full(n, band[0])
        frame = pd.DataFrame(
            {
                "issue_time": issue,
                "lead": lead,
                "target_end": issue + pd.to_timedelta(lead * 5, unit="min"),
                "actual": rng.normal(5000, 700, n),
                "stored": (anchor + raw).astype(np.float32).astype(np.float64),
                "forecast": anchor + raw,
                "raw": raw,
                "anchor": anchor,
                "base": base,
                **{x_col(c): X[:, j] for j, c in enumerate(columns)},
                **{shap_col(c): shap[:, j] for j, c in enumerate(columns)},
            }
        )
        frame.to_parquet(tmp_path / FITS_DIR / f"{key}.parquet", index=False)
        gain = dict(zip(columns, rng.random(len(columns)).tolist(), strict=True))
        fits.append(
            {
                "key": key,
                "retrain": retrain,
                "band": list(band),
                "target_form": form,
                "n_trees": 10,
                "num_leaves": 31,
                "eligible_pairs": n,
                "gain": gain,
                "gate": {"n": n},
            }
        )
    manifest = {"variant": "C", "columns": columns, "fits": fits, "gate": {"passed": True}}
    (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    return StoredSample(tmp_path, manifest)


def test_summaries_pool_bands_and_add_the_anchor(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    s = summarise(sample, SC, "Australia/Melbourne")
    assert s.bands == [(1, 6), (73, 288)]
    assert list(s.family_abs.columns) == ["5–30 min", "6–24 h"]
    assert s.pairs_by_band == {(1, 6): 800, (73, 288): 800}
    # Latest fits: dem_now carries the anchor's spread (≈ 200 MW sd), so it outweighs a
    # raw contribution (≈ 50 MW sd); in the raw table by form it does not stand out.
    assert s.feature_abs.loc["dem_now", "5–30 min"] > 3 * s.feature_abs.loc["wx_temp", "5–30 min"]
    by_form = s.family_abs_by_form.set_index(["band", "form"])
    assert set(by_form.index) == {
        ("5–30 min", "latest"),
        ("6–24 h", "direct"),
        ("6–24 h", "residual"),
    }
    assert by_form.loc[("6–24 h", "direct"), "fits"] == 1
    # gain shares sum to 1 per band
    np.testing.assert_allclose(s.gain_family.sum(axis=0), 1.0)
    # one stability pair per band, with the form change shown
    st = s.stability.set_index("band")
    assert st.loc["6–24 h", "form_a"] == "direct" and st.loc["6–24 h", "form_b"] == "residual"
    assert -1 <= st.loc["5–30 min", "spearman"] <= 1
    # question 5: terms add up to the forecast in every group
    q = s.question5
    for _, g in q.groupby(["basis", "group"]):
        parts = g[~g["term"].isin(["forecast_total"])]
        total = g[g["term"] == "forecast_total"].iloc[0]
        assert parts["diff"].sum() == pytest.approx(total["diff"], abs=1e-6)
        assert total["n_top"] == pytest.approx(0.05 * total["n_all"], abs=1)
    assert set(q["group"]) == {"overall", "summer", "winter"}
    # the overall row's forecast_total equals the stored forecasts' mean
    allrow = q[(q.basis == "actual") & (q.group == "overall") & (q.term == "forecast_total")]
    frames = [sample.frame(f["key"]) for f in sample.fits if f["band"] == [73, 288]]
    stored = np.concatenate([f["stored"].to_numpy() for f in frames])
    assert allrow["all"].iloc[0] == pytest.approx(stored.mean(), abs=1e-3)


def test_band_labels() -> None:
    assert [band_label(b) for b in CFG.ml.lead_bands] == [
        "5–30 min",
        "35 min–2 h",
        "2–6 h",
        "6–24 h",
    ]


def test_fixed_report_text_has_no_banned_phrase() -> None:
    from gridcast.anomalies.evidence import banned_in

    phrases = CFG.anomalies.evidence.banned_phrases
    assert banned_in(_how_to_read() + _disclosures(CFG), phrases) == []
    assert issubclass(BannedPhraseError, RuntimeError)
