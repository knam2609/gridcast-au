"""The frozen anomaly recipe's own rules (m4_plan.md step 4): hash, splits, case studies, cuts.

This module is frozen with the detector modules (:data:`gridcast.anomalies.freeze.FROZEN_PATHS`),
so none of these rules can change after the development results are seen:

* :func:`anomaly_config_hash`: the whole ``anomalies`` config section (reference-event
  rules, every detector setting, injection, evidence, SHAP, case studies and cuts), minus
  output directories and the freeze record, plus the split dates. It is v1's hash;
  :func:`recipe_config_hash` adds ``anomalies_v2`` for v2 (m4_plan.md §2.10);
* :func:`recipe_for`: each version's :class:`~gridcast.anomalies.detectors.Recipe`;
* :func:`dev_split` and :func:`holdout_split`: the scored periods (§2.1);
* :func:`case_studies`: §3.2's rule with the step-4 tie rule (user 2026-11-19). The
  ``cases.n`` events with the lowest p from a calibrated detector, ties by more calibrated
  detectors firing, then a seeded draw. Each named NEM day (development only) adds the
  event that ranks first among the calibrated events whose span touches that day;
* :func:`cut_indices`, :func:`truncated_detection` and :func:`differences`: A5 on real data
  (user 2026-11-19). The run is repeated on the series truncated at seeded cuts inside the
  scored period, and everything at or before each cut must be identical;
* :func:`forest_leaks`: §2.8 test 3 on the run's own forests.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gridcast.anomalies.detectors import LINES, ROBUST_DISTANCE, Recipe, v1_recipe
from gridcast.anomalies.events import Event
from gridcast.anomalies.run import Detection, Split, detect
from gridcast.anomalies.series import MarketSeries, QuantileBand, nem_day_ordinals
from gridcast.config import AnomaliesConfig, AppConfig

# Not hashed: where runs are stored, and the freeze record itself.
HASH_EXCLUDED: dict[str, Any] = {
    "output_dir": True,
    "frozen_config_hash": True,
    "frozen_commit": True,
    "explain": {"output_dir"},
}
V2_HASH_EXCLUDED: dict[str, Any] = {"frozen_config_hash": True, "frozen_commit": True}
VERSIONS = (1, 2)
# Seed streams of anomalies.seed (injections use 1-7).
CASE_SEED_INDEX = 101
CUT_SEED_INDEX = 102


def anomaly_config_hash(cfg: AppConfig) -> str:
    """Hash of every setting that changes an anomaly run (not paths or the freeze record)."""
    payload = {
        "anomalies": cfg.anomalies.model_dump(mode="json", exclude=HASH_EXCLUDED),
        "splits": {
            "holdout_start": str(cfg.backtest.holdout_start),
            "holdout_end": str(cfg.backtest.holdout_end),
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def recipe_config_hash(cfg: AppConfig, version: int) -> str:
    """v1: :func:`anomaly_config_hash`. v2: the same payload plus ``anomalies_v2`` (minus
    its freeze record)."""
    if version == 1:
        return anomaly_config_hash(cfg)
    if version != 2:
        raise ValueError(f"no anomaly recipe version {version}")
    payload = {
        "anomalies": cfg.anomalies.model_dump(mode="json", exclude=HASH_EXCLUDED),
        "anomalies_v2": cfg.anomalies_v2.model_dump(mode="json", exclude=V2_HASH_EXCLUDED),
        "splits": {
            "holdout_start": str(cfg.backtest.holdout_start),
            "holdout_end": str(cfg.backtest.holdout_end),
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def recipe_for(cfg: AppConfig, version: int) -> Recipe:
    """v1: the recipe frozen in step 4. v2 (§2.10): D4 with ``anomalies_v2.price_level``,
    D5 keeping ``anomalies.price_level``, and D6 the robust distance."""
    if version == 1:
        return v1_recipe(cfg.anomalies)
    if version != 2:
        raise ValueError(f"no anomaly recipe version {version}")
    v2 = cfg.anomalies_v2
    return Recipe(
        2,
        v2.price_level.scale_aud,
        cfg.anomalies.price_level.scale_aud,
        ROBUST_DISTANCE,
        v2.robust_distance,
    )


# --------------------------------------------------------------------------- splits


def dev_split(cfg: AppConfig) -> Split:
    """Development: market detectors and lines from ``market_first_day``, D2/D3 from
    ``residual_first_day``, both to ``backtest.holdout_start`` (exclusive)."""
    a = cfg.anomalies
    return Split("dev", a.market_first_day, a.residual_first_day, cfg.backtest.holdout_start)


def holdout_split(cfg: AppConfig) -> Split:
    """Holdout: every detector from ``holdout_start`` to ``holdout_end`` (exclusive)."""
    if cfg.backtest.holdout_end is None:
        raise ValueError("backtest.holdout_end is not set")
    start = cfg.backtest.holdout_start
    return Split("holdout", start, start, cfg.backtest.holdout_end)


# --------------------------------------------------------------------------- case studies


@dataclass(frozen=True, slots=True)
class CasePick:
    reason: str  # "lowest p (rank k)" or "named day YYYY-MM-DD"
    event: int | None  # position in the event list; None: no calibrated event that day
    also_named: tuple[dt.date, ...] = ()  # named days this rule pick also covers


def case_order(events: Sequence[Event], seed: int) -> list[int]:
    """Positions of the calibrated events (min p not None), ranked: lowest p, then more
    calibrated detectors firing, then a seeded draw (one uniform key per event, in order)."""
    keys = np.random.default_rng([seed, CASE_SEED_INDEX]).random(len(events))
    ranked = [
        (e.min_p, -sum(d not in LINES for d in e.detectors), float(keys[i]), i)
        for i, e in enumerate(events)
        if e.min_p is not None
    ]
    return [i for *_, i in sorted(ranked)]


def _touches(event: Event, day: dt.date, days: np.ndarray) -> bool:
    return bool(days[event.first] <= day.toordinal() <= days[event.last])


def case_studies(
    events: Sequence[Event],
    ends: pd.DatetimeIndex,
    cfg: AnomaliesConfig,
    named: bool,
) -> list[CasePick]:
    """§3.2's case studies: ``cases.n`` by rank, then each named day (if ``named``)."""
    order = case_order(events, cfg.seed)
    days = nem_day_ordinals(ends)
    named_days = list(cfg.cases.named_days) if named else []
    first_of_day: dict[dt.date, int | None] = {}
    for day in named_days:
        first_of_day[day] = next((i for i in order if _touches(events[i], day, days)), None)
    picks = []
    for rank, i in enumerate(order[: cfg.cases.n], start=1):
        covered = tuple(d for d, j in first_of_day.items() if j == i)
        picks.append(CasePick(f"lowest p (rank {rank})", i, covered))
    chosen = {p.event for p in picks}
    for day, j in first_of_day.items():
        if j is None or j not in chosen:
            picks.append(CasePick(f"named day {day.isoformat()}", j))
    return picks


# --------------------------------------------------------------------------- A5 on real data


def cut_indices(ends: pd.DatetimeIndex, split: Split, cfg: AnomaliesConfig) -> np.ndarray:
    """``causality.real_cuts`` seeded interval indices, uniform over the split's scored
    period (from its earlier first day), never the last interval."""
    start, stop = split.index_range(ends)
    candidates = np.arange(start, stop - 1)
    k = cfg.causality.real_cuts
    if len(candidates) < k:
        raise ValueError("the scored period is too short for the cuts")
    rng = np.random.default_rng([cfg.seed, CUT_SEED_INDEX])
    return np.sort(rng.choice(candidates, size=k, replace=False))


def truncated_detection(
    series: MarketSeries,
    band: QuantileBand | None,
    split: Split,
    cfg: AnomaliesConfig,
    cut: int,
    recipe: Recipe | None = None,
) -> Detection:
    """The detection from data ending at interval ``cut`` (forecasts issued after it dropped)."""
    end = series.ends[cut]
    return detect(
        series.head(cut + 1),
        None if band is None else band.issued_until(end),
        split,
        cfg,
        recipe=recipe,
    )


def differences(full: Detection, cut_run: Detection, cut: int) -> list[str]:
    """What differs at or before interval ``cut``: scores, p-values, window sizes, tiers
    (lines included) and event openings."""
    out = []
    upto = slice(0, cut + 1)
    for d in full.run.recipe.detectors:
        for label, x, y in (
            ("score", full.run.scores[d], cut_run.run.scores[d]),
            ("p", full.run.pvalues[d], cut_run.run.pvalues[d]),
            ("n_window", full.run.n_window[d], cut_run.run.n_window[d]),
        ):
            if not np.array_equal(x[upto], y[upto], equal_nan=True):
                out.append(f"{d} {label}")
    for d, t in full.tiers.items():
        if not np.array_equal(t[upto], cut_run.tiers[d][upto]):
            out.append(f"{d} tier")
    opened_full = [e.first for e in full.events if e.first <= cut]
    opened_cut = [e.first for e in cut_run.events if e.first <= cut]
    if opened_full != opened_cut:
        out.append("event openings")
    return out


def forest_leaks(detection: Detection) -> list[str]:
    """§2.8 test 3: months whose model (a forest, or v2's distance) saw an interval ending
    after the month's start."""
    return [
        f"{mm.month // 12}-{mm.month % 12 + 1:02d}"
        for mm in detection.run.state.month_fits()
        if mm.train_last_end > mm.start
    ]
