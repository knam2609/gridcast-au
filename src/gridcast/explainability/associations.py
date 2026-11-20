"""Descriptive price associations (m4_plan.md §2.7; §1 "Clarified in step 5"). Association
only: a rate is *more frequent when* a condition holds, never more.

* **Rates:** P(price > $300/MWh) and P(price < $0/MWh), by bins of observables from the same
  5-minute interval: demand, ERA5 population-weighted temperature (the value the evidence
  uses), wind and solar cleared output, net interchange, local hour and local month, and
  one two-way table (demand tercile × wind tercile).
* **Period:** NEM days by interval start, from ``anomalies.market_first_day`` to
  ``backtest.holdout_start`` (the market development period).
* **Bins** (user decision 2026-11-20): edges from development data, reused unchanged by the
  holdout (step 6). Deciles are right-closed bins between the 10% … 90% quantiles. Solar has
  a *zero output* bin plus deciles of positive output. Temperature bins are fixed,
  [5k, 5k + 5) °C. Hour and month are categories.
* **Uncertainty:** percentile intervals from day-block bootstrap resamples (NEM days), one
  seeded set of resamples shared by every table.
* An interval that lacks a variable is left out of that variable's table only.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig, PriceAssociationsConfig
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.timeutils import INTERVAL, NEM_TZ

QUANTILE_VARIABLES = ("demand", "wind", "net_interchange")
VARIABLES = ("demand", "temperature", "wind", "solar", "net_interchange", "hour", "month")
TITLES = {
    "demand": "Demand (MW)",
    "temperature": "ERA5 population-weighted temperature (°C)",
    "wind": "Wind output, semi-scheduled (MW)",
    "solar": "Solar output, semi-scheduled (MW)",
    "net_interchange": "Net interchange (MW, positive = export)",
    "hour": "Local hour",
    "month": "Local month",
}


@dataclass(slots=True)
class AssociationData:
    """One row per 5-minute interval of the period."""

    ends: pd.DatetimeIndex  # UTC interval ends
    day: np.ndarray  # NEM day of the interval start, as a day ordinal
    price: np.ndarray
    observed: dict[str, np.ndarray]  # float, NaN where missing (hour, month: int)

    def __post_init__(self) -> None:
        n = len(self.ends)
        if self.day.shape != (n,) or self.price.shape != (n,):
            raise ValueError("one day and one price per interval")
        for k, v in self.observed.items():
            if v.shape != (n,):
                raise ValueError(f"{k} must have one value per interval")
        if np.isnan(self.price).any():
            raise ValueError("every interval of the period needs a price")


def period_bounds(cfg: AppConfig) -> tuple[dt.date, dt.date]:
    """First and last NEM day of the market development period."""
    return cfg.anomalies.market_first_day, cfg.backtest.holdout_start - dt.timedelta(days=1)


def nem_days(ends: pd.DatetimeIndex) -> np.ndarray:
    """NEM day ordinal of each interval's start."""
    starts = (ends - INTERVAL).tz_convert(NEM_TZ)
    return np.asarray([d.toordinal() for d in starts.date], dtype=np.int64)


def load_association_data(
    engine: Engine, cfg: AppConfig, region_id: str, first: dt.date, last: dt.date
) -> AssociationData:
    """The period's intervals (NEM days ``first`` … ``last`` by interval start)."""
    from gridcast.anomalies.inputs import load_market_series
    from gridcast.features.inputs import load_context, load_feature_builder
    from gridcast.features.weather import select_observed

    end = nem_midnight_utc(str(last + dt.timedelta(days=1)))
    series = load_market_series(engine, cfg, region_id, last_end=end)
    days = nem_days(series.ends)
    keep = (days >= first.toordinal()) & (days <= last.toordinal())
    ends = series.ends[keep]
    if len(ends) != (last - first).days * 288 + 288:
        raise ValueError("the period is not a complete 5-minute grid")

    ctx = load_context(engine, cfg, region_id)
    builder = load_feature_builder(engine, cfg, region_id, ctx=ctx)
    if builder.weather is None:
        raise ValueError("the builder has no weather")
    pos = np.asarray(builder.ctx.times.get_indexer(ends), dtype=np.int64)
    temp = np.full(len(ends), np.nan)
    on_grid = pos >= 0
    temp[on_grid] = select_observed(builder.weather, ["temp_weighted"], pos[on_grid])[
        "temp_weighted"
    ]
    local = (ends - INTERVAL).tz_convert(series.tz)
    values = {
        "demand": series.demand[keep].astype(np.float64),
        "temperature": temp.astype(np.float64),
        "wind": series.wind_cleared[keep].astype(np.float64),
        "solar": series.solar_cleared[keep].astype(np.float64),
        "net_interchange": series.net_interchange[keep].astype(np.float64),
        "hour": np.asarray(local.hour, dtype=np.int64),
        "month": np.asarray(local.month, dtype=np.int64),
    }
    return AssociationData(ends, days[keep], series.price[keep].astype(np.float64), values)


# --------------------------------------------------------------------------- bins


@dataclass(frozen=True, slots=True)
class Binning:
    """Bins of one variable; :meth:`codes` gives −1 where the value is missing."""

    variable: str
    kind: str  # "quantile", "zero_quantile", "fixed" or "category"
    edges: tuple[float, ...] = ()  # quantile: interior edges; zero_quantile: of positives
    width: float = 0.0  # fixed
    categories: tuple[int, ...] = ()  # category

    @property
    def n_bins(self) -> int:
        if self.kind == "quantile":
            return len(self.edges) + 1
        if self.kind == "zero_quantile":
            return len(self.edges) + 2
        if self.kind == "category":
            return len(self.categories)
        raise ValueError("fixed bins are open-ended; see fixed_codes")

    def codes(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        out = np.full(len(v), -1, dtype=np.int64)
        ok = np.isfinite(v)
        if self.kind == "quantile":
            out[ok] = np.searchsorted(np.asarray(self.edges), v[ok], side="left")
        elif self.kind == "zero_quantile":
            if np.any(v[ok] < 0):
                raise ValueError(f"{self.variable}: negative values")
            zero = ok & (v == 0)
            pos = ok & (v > 0)
            out[zero] = 0
            out[pos] = 1 + np.searchsorted(np.asarray(self.edges), v[pos], side="left")
        elif self.kind == "category":
            lookup = {c: i for i, c in enumerate(self.categories)}
            out[ok] = [lookup.get(int(x), -1) for x in v[ok]]
        else:
            raise ValueError("fixed bins: use fixed_codes")
        return out

    def labels(self, unit: str = "") -> list[str]:
        u = f" {unit}" if unit else ""
        if self.kind == "category":
            return [str(c) for c in self.categories]
        edges = [f"{e:,.0f}" for e in self.edges]
        if self.kind == "quantile":
            return (
                [f"≤ {edges[0]}{u}"]
                + [f"{a} – {b}{u}" for a, b in itertools.pairwise(edges)]
                + [f"> {edges[-1]}{u}"]
            )
        if self.kind == "zero_quantile":
            return (
                [f"0{u} (zero output)", f"0 – {edges[0]}{u}"]
                + [f"{a} – {b}{u}" for a, b in itertools.pairwise(edges)]
                + [f"> {edges[-1]}{u}"]
            )
        raise ValueError("fixed bins are labelled by fixed_codes")


def quantile_edges(values: np.ndarray, k: int) -> tuple[float, ...]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return tuple(float(e) for e in np.quantile(v, np.arange(1, k) / k))


def fixed_codes(values: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Codes 0… over the occupied fixed bins [width·k, width·(k + 1)), and their k."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    k = np.floor(v[ok] / width).astype(np.int64)
    present = np.unique(k)
    out = np.full(len(v), -1, dtype=np.int64)
    out[ok] = np.searchsorted(present, k)
    return out, present


def development_binnings(data: AssociationData, cfg: PriceAssociationsConfig) -> dict[str, Binning]:
    """Every variable's bins, from the development data (step 6 reuses them)."""
    out = {
        v: Binning(v, "quantile", quantile_edges(data.observed[v], cfg.quantile_bins))
        for v in QUANTILE_VARIABLES
    }
    solar = data.observed["solar"]
    out["solar"] = Binning(
        "solar", "zero_quantile", quantile_edges(solar[solar > 0], cfg.quantile_bins)
    )
    out["temperature"] = Binning("temperature", "fixed", width=cfg.temperature_bin_c)
    out["hour"] = Binning("hour", "category", categories=tuple(range(24)))
    out["month"] = Binning("month", "category", categories=tuple(range(1, 13)))
    for v in ("demand", "wind"):
        out[f"{v}_two_way"] = Binning(
            v, "quantile", quantile_edges(data.observed[v], cfg.two_way_bins)
        )
    return out


# --------------------------------------------------------------------------- rates


@dataclass(slots=True)
class Bootstrap:
    """One seeded set of day-block resamples: weights[b, d] = times day d is drawn."""

    days: np.ndarray  # sorted unique day ordinals
    weights: np.ndarray  # (B, D) float64
    level: float

    @classmethod
    def draw(cls, day: np.ndarray, resamples: int, seed: Sequence[int], level: float) -> Bootstrap:
        days = np.unique(day)
        rng = np.random.default_rng(list(seed))
        draws = rng.integers(0, len(days), size=(resamples, len(days)))
        weights = np.zeros((resamples, len(days)), dtype=np.float64)
        for b in range(resamples):
            weights[b] = np.bincount(draws[b], minlength=len(days))
        return cls(days, weights, level)


@dataclass(slots=True)
class RateTable:
    variable: str
    labels: list[str]
    frame: pd.DataFrame  # per bin: n, share, high, high_lo, high_hi, low, low_lo, low_hi
    missing: int  # intervals without the variable
    extra: dict[str, object] = field(default_factory=dict)


def bin_rates(
    data: AssociationData,
    codes: np.ndarray,
    n_bins: int,
    boot: Bootstrap,
    cfg: PriceAssociationsConfig,
) -> pd.DataFrame:
    """Per bin: interval count, share, and both rates with bootstrap intervals."""
    d = np.searchsorted(boot.days, data.day)
    ok = codes >= 0
    counts = np.zeros((len(boot.days), n_bins))
    np.add.at(counts, (d[ok], codes[ok]), 1.0)
    n = counts.sum(axis=0)
    out: dict[str, np.ndarray] = {"n": n, "share": n / max(int(ok.sum()), 1)}
    boot_n = boot.weights @ counts
    q = 100 * np.array([(1 - boot.level) / 2, (1 + boot.level) / 2])
    for name, hit in (
        ("high", data.price > cfg.high_price),
        ("low", data.price < cfg.low_price),
    ):
        hits = np.zeros_like(counts)
        sel = ok & hit
        np.add.at(hits, (d[sel], codes[sel]), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = hits.sum(axis=0) / n
            boot_rate = (boot.weights @ hits) / boot_n
        lo, hi = (np.full(n_bins, np.nan) for _ in range(2))
        has = n > 0
        lo[has], hi[has] = np.nanpercentile(boot_rate[:, has], q, axis=0)
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return pd.DataFrame(out)


def rate_table(
    data: AssociationData,
    variable: str,
    binning: Binning,
    boot: Bootstrap,
    cfg: PriceAssociationsConfig,
    unit: str = "",
) -> RateTable:
    values = data.observed[variable]
    if binning.kind == "fixed":
        codes, present = fixed_codes(values, binning.width)
        w = binning.width
        labels = [
            f"{w * k:,.0f} to {w * (k + 1):,.0f}{' ' + unit if unit else ''}" for k in present
        ]
        frame = bin_rates(data, codes, len(present), boot, cfg)
        frame.insert(0, "lower", w * present)
    else:
        codes = binning.codes(values)
        labels = binning.labels(unit)
        frame = bin_rates(data, codes, binning.n_bins, boot, cfg)
    frame.insert(0, "bin", labels)
    return RateTable(variable, labels, frame, int((codes < 0).sum()))


def two_way_table(
    data: AssociationData,
    demand: Binning,
    wind: Binning,
    boot: Bootstrap,
    cfg: PriceAssociationsConfig,
) -> pd.DataFrame:
    """Demand tercile × wind tercile, one row per cell."""
    a = demand.codes(data.observed["demand"])
    b = wind.codes(data.observed["wind"])
    k = wind.n_bins
    codes = np.where((a >= 0) & (b >= 0), a * k + b, -1)
    frame = bin_rates(data, codes, demand.n_bins * k, boot, cfg)
    names = ("low", "middle", "high") if k == 3 else tuple(str(i + 1) for i in range(k))
    frame.insert(0, "wind", [names[i % k] for i in range(len(frame))])
    frame.insert(0, "demand", [names[i // k] for i in range(len(frame))])
    return frame


@dataclass(slots=True)
class Associations:
    first: dt.date
    last: dt.date
    n: int
    overall: pd.DataFrame  # one row: high, low with intervals
    binnings: dict[str, Binning]
    tables: dict[str, RateTable]
    two_way: pd.DataFrame
    resamples: int
    days: int


UNITS = {"demand": "MW", "temperature": "°C", "wind": "MW", "solar": "MW", "net_interchange": "MW"}


def compute_associations(
    data: AssociationData,
    cfg: PriceAssociationsConfig,
    seed: int,
    first: dt.date,
    last: dt.date,
    binnings: dict[str, Binning] | None = None,
) -> Associations:
    """Every table; ``binnings`` defaults to the development bins of ``data`` itself."""
    bins = binnings if binnings is not None else development_binnings(data, cfg)
    boot = Bootstrap.draw(data.day, cfg.bootstrap_resamples, (seed, cfg.seed_stream), cfg.ci_level)
    overall = bin_rates(data, np.zeros(len(data.day), np.int64), 1, boot, cfg)
    tables = {v: rate_table(data, v, bins[v], boot, cfg, UNITS.get(v, "")) for v in VARIABLES}
    two_way = two_way_table(data, bins["demand_two_way"], bins["wind_two_way"], boot, cfg)
    return Associations(
        first,
        last,
        len(data.day),
        overall,
        bins,
        tables,
        two_way,
        cfg.bootstrap_resamples,
        len(boot.days),
    )
