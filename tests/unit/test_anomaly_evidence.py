"""Event evidence, summaries and A4's checks (m4_plan.md §2.5, step 3b), on synthetic data."""

from __future__ import annotations

import dataclasses
import json
import re
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gridcast.anomalies.detectors import (
    DEMAND_LEVEL,
    ISOLATION_FOREST,
    LINE_NOTABLE,
    PRICE_LEVEL,
    PRICE_VOLATILITY,
    RESIDUAL_DAYAHEAD,
    RESIDUAL_SHORT,
)
from gridcast.anomalies.events import merge_events
from gridcast.anomalies.evidence import (
    KNOWN,
    REQUIRED_FIELDS,
    RETRO,
    AemoRun,
    EvidenceContext,
    PointFact,
    ShapRow,
    banned_in,
    build_evidence,
    demand_peak,
    epoch_minutes,
    event_pairs,
    fact,
    missing_fields,
    pair_for,
    peak_detector,
    shap_pairs,
    summarise,
    unavailable,
)
from gridcast.anomalies.run import Detection, detect
from gridcast.anomalies.series import MarketSeries, QuantileBand
from gridcast.explainability.families import families
from tests.anomaly_fixtures import base_inputs, small_config, split


def _sentences(text: str) -> int:
    return len(re.findall(r"\.(?:\s|$)", text))


@cache
def _detection() -> Detection:
    series, band = base_inputs()
    return detect(series, band, split(), small_config())


def _band_fact(band: QuantileBand, key: tuple[int, int]) -> PointFact | None:
    minutes = band.issue_times.as_unit("ns").asi8 // (60 * 10**9)
    row = int(np.searchsorted(minutes, key[0]))
    if row >= len(minutes) or minutes[row] != key[0] or not band.valid[row, key[1] - 1]:
        return None
    lo, hi = float(band.lower[row, key[1] - 1]), float(band.upper[row, key[1] - 1])
    return PointFact((lo + hi) / 2, lo, hi)


def _shap_row(forecast: float) -> ShapRow:
    fam = families("C")
    contrib = np.linspace(-50, 50, len(fam.columns))
    sums = fam.sums(contrib[None, :])[0]
    return ShapRow(
        fit="2024-02-01_h001-006",
        target_form="latest",
        anchor=forecast - 100.0 - float(contrib.sum()),
        base=100.0,
        forecast=forecast,
        families=tuple(
            (k, t, float(v)) for k, t, v in zip(fam.keys, fam.titles, sums, strict=True)
        ),
        features=tuple((n, float(v)) for n, v in zip(fam.columns, contrib, strict=True)),
    )


def _context(series: MarketSeries, band: QuantileBand, events, detection) -> EvidenceContext:
    cfg = small_config()
    n = len(series)
    pairs = {k for e in events for k in event_pairs(e, detection, cfg)}
    peaks = {k for e in events for k in shap_pairs(e, detection, cfg)}
    point, missing = {}, {}
    for key in pairs | peaks:
        pf = _band_fact(band, key)
        if pf is None:
            missing[key] = "outside the synthetic mask"
        else:
            point[key] = pf
    aemo = {}
    for e in events:
        end = series.ends[e.peak]
        issue, _ = pair_for(end, cfg.residual.short_leads[0])
        end_min = epoch_minutes(end)
        run_ts = pd.Timestamp(issue * 60 * 10**9, tz="UTC")
        aemo[("p5min", issue, end_min)] = AemoRun(
            float(series.demand[e.peak]) + 10, run_ts - pd.Timedelta(minutes=1)
        )
        aemo[("predispatch", issue, end_min + (-end_min) % 30)] = AemoRun(
            5000.0, run_ts + pd.Timedelta(minutes=10)
        )
    apc = np.zeros(n, dtype=np.int64)
    apc[n // 3 : n // 3 + 20] = 1
    battery = np.zeros(n)
    battery[n // 2 :] = 50.0
    avail = np.full(n, np.nan)
    avail[n // 2 :] = series.wind_uigf[n // 2 :]
    return EvidenceContext(
        series=series,
        cfg=cfg,
        apc_bits=apc,
        suspended=np.zeros(n, dtype=np.int64),
        wind_available=avail,
        solar_available=avail.copy(),
        battery_gen=battery,
        battery_load=np.zeros(n),
        point=point,
        point_missing=missing,
        gfs={k: 21.5 for k in pairs},
        era5={e.peak: 22.0 for e in events},
        aemo=aemo,
        aemo_first_run={"p5min": series.ends[0], "predispatch": series.ends[0]},
        shap={k: _shap_row(point[k].forecast) for k in peaks if k in point},
        shap_missing={k: missing[k] for k in peaks if k in missing},
        point_source="synthetic",
    )


# --------------------------------------------------------------------------- end to end


def test_every_synthetic_event_has_complete_clean_evidence() -> None:
    series, band = base_inputs()
    det = _detection()
    assert len(det.events) > 20
    ctx = _context(series, band, det.events, det)
    peak_kinds = set()
    for event in det.events:
        ev = build_evidence(event, det, ctx)
        assert missing_fields(ev) == [], event
        json.dumps(ev, allow_nan=False)  # jsonb-ready: no NaN anywhere
        text = summarise(ev)
        assert _sentences(text) == 3, text
        assert banned_in(text, small_config().evidence.banned_phrases) == [], text
        assert "nan" not in text.lower()
        for path in REQUIRED_FIELDS:
            node = ev
            for part in path.split("."):
                node = node[part]
            assert ("unavailable" in node) or node["known"] in (KNOWN, RETRO), path
        peak_kinds.add(peak_detector(event, det))
        if demand_peak(event, det) is None:
            assert "unavailable" in ev["shap"]["latest"]
    assert {DEMAND_LEVEL, RESIDUAL_SHORT, PRICE_LEVEL} <= peak_kinds


def test_the_profile_median_is_the_detectors() -> None:
    series, band = base_inputs()
    det = _detection()
    ctx = _context(series, band, det.events, det)
    d1 = [e for e in det.events if peak_detector(e, det) == DEMAND_LEVEL]
    assert d1
    for e in d1:
        ev = build_evidence(e, det, ctx)
        median = ev["values"]["demand_profile"]["value"]["median"]
        assert median == pytest.approx(det.run.state.demand_profile.median[e.peak], abs=1e-9)


# --------------------------------------------------------------------------- demand peak


def _fake_detection(series: MarketSeries, alerts: dict[str, dict[int, tuple[int, float, float]]]):
    n = len(series)
    tiers, pvalues, scores = {}, {}, {}
    for d, rows in alerts.items():
        t, p, s = np.zeros(n, np.int8), np.full(n, 0.5), np.zeros(n)
        for i, (tier, pp, ss) in rows.items():
            t[i], p[i], s[i] = tier, pp, ss
        tiers[d], pvalues[d], scores[d] = t, p, s
    events = merge_events(tiers, pvalues, scores, series.price, 0, n, 6)
    det = SimpleNamespace(
        tiers=tiers, run=SimpleNamespace(pvalues=pvalues, scores=scores, ends=series.ends)
    )
    return det, events


def test_a_mixed_event_explains_its_demand_peak() -> None:
    series, band = base_inputs()
    i = len(series) // 2 + 7
    det, events = _fake_detection(
        series, {DEMAND_LEVEL: {i: (1, 8e-4, 5.0)}, PRICE_LEVEL: {i + 2: (2, 5e-5, 9.0)}}
    )
    (event,) = events
    assert event.peak == i + 2 and peak_detector(event, det) == PRICE_LEVEL
    assert demand_peak(event, det) == i
    ctx = _context(series, band, events, det)
    ev = build_evidence(event, det, ctx)
    want = series.ends[i].tz_convert(series.tz).isoformat()
    assert ev["event"]["demand_peak"]["value"]["interval_end"]["local"] == want
    shap = ev["shap"]["latest"]
    if "value" in shap:
        assert shap["value"]["interval_end"]["local"] == want
        fams = sum(f["mw"] for f in shap["value"]["families"])
        assert shap["value"]["anchor"]["mw"] + shap["value"]["base_value"] + fams == pytest.approx(
            shap["value"]["forecast"]
        )
    # The forecasts block stays at the event peak.
    issue, _ = pair_for(series.ends[i + 2], 1)
    if "value" in ev["forecasts"]["latest"]:
        got = pd.Timestamp(ev["forecasts"]["latest"]["value"]["issue_time"]["local"])
        assert epoch_minutes(got.tz_convert("UTC")) == issue
    assert summarise(ev).startswith("Severe event")


def test_pairs_use_the_detectors_leads() -> None:
    series, _ = base_inputs()
    for end in series.ends[::997]:
        for first in (1, 283):
            issue, lead = pair_for(end, first)
            assert issue % 30 == 0 and first <= lead <= first + 5
            assert issue + 5 * lead == epoch_minutes(end)
    t = pd.Timestamp("2024-04-01 10:05", tz="UTC")
    assert pair_for(t, 1) == (epoch_minutes(pd.Timestamp("2024-04-01 10:00", tz="UTC")), 1)
    t = pd.Timestamp("2024-04-01 10:30", tz="UTC")
    assert pair_for(t, 1) == (epoch_minutes(pd.Timestamp("2024-04-01 10:00", tz="UTC")), 6)


# --------------------------------------------------------------------------- pieces


def test_aemo_labels_and_reasons() -> None:
    series, band = base_inputs()
    det = _detection()
    ctx = _context(series, band, det.events, det)
    mid = next(e for e in det.events if epoch_minutes(series.ends[e.peak]) % 30 != 0)
    ev = build_evidence(mid, det, ctx)
    p5 = ev["forecasts"]["aemo_p5min_latest"]
    assert p5["known"] == KNOWN and p5["value"]["error"] == pytest.approx(10.0)
    pd_latest = ev["forecasts"]["aemo_predispatch_latest"]
    assert pd_latest["known"] == RETRO  # the period mean needs intervals after the peak
    assert "reaches" in ev["forecasts"]["aemo_predispatch_dayahead"]["unavailable"]
    late = EvidenceContext(**{**_init_args(ctx), "aemo_first_run": {"p5min": None}})
    ev2 = build_evidence(mid, det, late)
    assert ev2["forecasts"]["aemo_p5min_latest"] == unavailable(
        "AEMO's p5min forecasts are not loaded"
    )


def _init_args(ctx: EvidenceContext) -> dict:
    return {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx) if f.init}


def test_battery_and_availability_start_when_the_data_do() -> None:
    series, band = base_inputs()
    det = _detection()
    ctx = _context(series, band, det.events, det)
    half = len(series) // 2
    early = next(e for e in det.events if e.peak < half)
    late = next(e for e in det.events if e.peak >= half)
    a, b = build_evidence(early, det, ctx), build_evidence(late, det, ctx)
    assert "non-zero bidirectional-unit" in a["supply"]["battery"]["unavailable"]
    assert "May 2024" in a["supply"]["wind_available"]["unavailable"]
    assert b["supply"]["battery"]["value"]["generation"] == 50.0
    assert b["supply"]["battery"]["note"] == "bidirectional units only"
    assert "value" in b["supply"]["wind_available"]


def test_flags_are_counted_over_the_span() -> None:
    series, band = base_inputs()
    n = len(series)
    det, events = _fake_detection(series, {DEMAND_LEVEL: {n // 3 + 5: (1, 5e-4, 6.0)}})
    ctx = _context(series, band, events, det)
    ev = build_evidence(events[0], det, ctx)
    span = ev["flags"]["span"]["value"]
    assert span["flagged_intervals"] == 1 and span["apc_bits"][0]["bit"] == 1
    assert "AEMO flagged 1 of 1 intervals: APC/APF binding in 1." in summarise(ev)


def test_banned_phrases_match_words_case_insensitively() -> None:
    phrases = small_config().evidence.banned_phrases
    assert banned_in("Prices rose Because demand fell", phrases) == ["because"]
    assert banned_in("This was due to heat", phrases) == ["due to"]
    assert banned_in("A becausee typo and causeway", phrases) == []
    assert banned_in("It Explains nothing", phrases) == ["explains"]


def test_missing_fields_catches_unlabelled_and_reasonless_fields() -> None:
    series, band = base_inputs()
    det = _detection()
    ctx = _context(series, band, det.events, det)
    ev = build_evidence(det.events[0], det, ctx)
    assert missing_fields(ev) == []
    ev["supply"]["margin"] = {"value": 5.0, "unit": "MW", "source": "x"}  # no timing label
    ev["weather"]["era5_temp"] = {"unavailable": " "}
    del ev["changes"]["price_5min"]
    assert missing_fields(ev) == ["changes.price_5min", "supply.margin", "weather.era5_temp"]
    with pytest.raises(ValueError, match="NaN"):
        fact(float("nan"), "MW", KNOWN, "x")
    with pytest.raises(ValueError, match="reason"):
        unavailable("")


# --------------------------------------------------------------------------- templates


def _evidence_for(detector: str) -> dict:
    local = "2024-12-09T15:40:00+11:00"
    times = {"local": local, "nem": "2024-12-09T14:40:00+10:00"}
    start = {"local": "2024-12-09T15:20:00+11:00", "nem": "x"}
    end = {"local": "2024-12-09T16:10:00+11:00", "nem": "x"}
    f = {
        "issue_time": {"local": "2024-12-09T15:30:00+11:00"},
        "forecast": 2679.0,
        "band_lower": 2500.0,
        "band_upper": 2900.0,
        "actual": 5518.0,
        "residual": 2839.0,
        "outside_band": True,
    }
    prof = {"median": 5000.0, "p10": 4500.0, "p90": 5500.0, "n": 120}
    return {
        "event": {
            "start": fact(start, None, KNOWN, "s"),
            "end": fact(end, None, KNOWN, "s"),
            "tier": fact("severe", None, KNOWN, "s"),
            "peak": fact(
                {"interval_end": times, "detector": detector, "p": 1.9e-5}, None, KNOWN, "s"
            ),
            "price_lines": fact(
                {
                    "intervals": 11,
                    "at_or_above_notable": 4,
                    "at_or_above_severe": 1,
                    "notable_aud": 300,
                    "severe_aud": 5000,
                    "max_price": 15500.0,
                },
                None,
                KNOWN,
                "s",
            ),
        },
        "values": {
            "demand": fact(5518.0, "MW", KNOWN, "s"),
            "price": fact(95.0, "$/MWh", KNOWN, "s"),
            "demand_profile": fact(prof, "MW", KNOWN, "s", day_type="business days"),
            "price_profile": fact(prof, "$/MWh", KNOWN, "s"),
            "price_volatility": fact(
                {"ratio": 0.16, "hour_min_price": -21.0, "hour_max_price": 95.0}, None, KNOWN, "s"
            ),
        },
        "forecasts": {"latest": fact(f, "MW", KNOWN, "s"), "dayahead": fact(f, "MW", KNOWN, "s")},
        "changes": {
            "price_5min": fact({"from": -21.0, "to": 95.0}, "$/MWh", KNOWN, "s"),
            "demand_5min": fact({"from": 5400.0, "to": 5518.0}, "MW", KNOWN, "s"),
        },
        "flags": {
            "span": fact(
                {"intervals": 11, "flagged_intervals": 0, "apc_bits": [], "suspended_intervals": 0},
                None,
                KNOWN,
                "s",
            )
        },
    }


@pytest.mark.parametrize(
    ("detector", "phrase"),
    [
        (DEMAND_LEVEL, "the median for that half-hour on the business days among the previous 28"),
        (RESIDUAL_SHORT, "2,839 MW above LightGBM C's latest forecast (2,679 MW, issued at 15:30)"),
        (RESIDUAL_DAYAHEAD, "day-ahead forecast (2,679 MW, issued at 15:30 on Mon 9 Dec 2024)"),
        (PRICE_LEVEL, "the price in the 5 minutes to 15:40 was $95/MWh"),
        (PRICE_VOLATILITY, "was 0.16 times its median over the previous 7 days"),
        (ISOLATION_FOREST, "a p-value of 1.9 × 10⁻⁵"),
        (LINE_NOTABLE, "4 of the event's 11 intervals were at or above $300/MWh and 1 at or above"),
    ],
)
def test_lead_sentence_by_peak_detector(detector: str, phrase: str) -> None:
    text = summarise(_evidence_for(detector))
    assert text.startswith("Severe event, 15:20–16:10 Mon 9 Dec 2024 (Melbourne time): ")
    assert phrase in text, text
    other = "the price moved from −$21 to $95/MWh" if "demand" in detector else "demand moved"
    assert other in text
    assert text.endswith("AEMO flagged nothing in the event's span.")
    assert _sentences(text) == 3
    assert banned_in(text, small_config().evidence.banned_phrases) == []


def test_wording_edge_cases() -> None:
    ev = _evidence_for(RESIDUAL_SHORT)
    ev["changes"]["price_5min"] = fact({"from": 7.0, "to": -0.3}, "$/MWh", KNOWN, "s")
    assert "from $7 to $0/MWh" in summarise(ev)  # no negative zero
    ev = _evidence_for(PRICE_VOLATILITY)
    ev["values"]["price_volatility"]["value"].update(hour_min_price=19.1, hour_max_price=19.4)
    assert "ranged from $19.10 to $19.40/MWh" in summarise(ev)
    ev["values"]["price_volatility"]["value"].update(hour_min_price=19.0, hour_max_price=19.0)
    assert "the price stayed at $19/MWh" in summarise(ev)
    ev["values"]["price_volatility"]["value"].update(hour_min_price=19.18, hour_max_price=19.183)
    assert "the price stayed at $19.18/MWh (to the cent)" in summarise(ev)
    ev = _evidence_for(LINE_NOTABLE)
    ev["event"]["price_lines"]["value"].update(intervals=1, at_or_above_notable=1)
    assert "; the event's single interval was at or above $300/MWh." in summarise(ev)
    ev["event"]["price_lines"]["value"].update(intervals=4, at_or_above_severe=0)
    assert "1 of the event's 4 intervals was at or above $300/MWh." in summarise(ev)


# --------------------------------------------------------------------------- stored lookups


def _wide(path, times: pd.DatetimeIndex, values: np.ndarray) -> None:
    frame = pd.DataFrame(values.astype(np.float32), columns=[f"h{h:03d}" for h in range(1, 7)])
    frame.insert(0, "origin_time", times)
    frame.to_parquet(path, index=False)


def test_point_lookup_gives_values_or_reasons(tmp_path) -> None:
    from gridcast.anomalies.evidence_inputs import OUTSIDE_MASK, lookup_points
    from gridcast.config import load_config
    from gridcast.forecasting.origins import nem_midnight_utc

    base = load_config()
    start = nem_midnight_utc("2022-11-01")
    times = pd.date_range(start, periods=4, freq="30min")
    forecast = 5000 + np.arange(24, dtype=np.float64).reshape(4, 6)
    other = forecast.copy()
    other[1, 2] = np.nan  # another model missing: outside M2's common mask
    back = tmp_path / "backtests"
    back.mkdir()
    for name, values in (("actual", forecast), ("lgbm_C", forecast), ("ridge_C", other)):
        _wide(back / f"demand__{name}__B.parquet", times, values)
    qr = tmp_path / "intervals" / "Q"
    qr.mkdir(parents=True)
    _wide(qr / "qr_lgbm_C__q025.parquet", times, forecast - 100)
    _wide(qr / "qr_lgbm_C__q975.parquet", times, forecast + 100)
    cfg = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={"data_root": tmp_path, "backtest_dir": Path("backtests")}
            ),
            "intervals": base.intervals.model_copy(update={"output_dir": Path("intervals")}),
            "anomalies": base.anomalies.model_copy(
                update={
                    "explain": base.anomalies.explain.model_copy(update={"dev_batch": "B"}),
                    "residual": base.anomalies.residual.model_copy(update={"dev_qr_batch": "Q"}),
                }
            ),
        }
    )
    m0 = int(times[0].value // (60 * 10**9))
    keys = [(m0, 1), (m0 + 30, 3), (m0 - 30, 2), (m0 + 90, 6)]
    found, missing, label = lookup_points(cfg, "dev", keys)
    assert found[(m0, 1)] == PointFact(5000.0, 4900.0, 5100.0)
    assert found[(m0 + 90, 6)].forecast == 5023.0
    assert missing[(m0 + 30, 3)] == OUTSIDE_MASK
    assert "start at issue time 2022-11-01 00:00 NEM time" in missing[(m0 - 30, 2)]
    assert "batch B" in label and "QR batch Q" in label
