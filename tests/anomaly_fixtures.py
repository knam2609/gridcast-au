"""Synthetic inputs for the M4 detector tests (no database).

The series crosses Melbourne's DST fall-back (7 Apr 2024) and two NEM-month starts, so
the profiles see a local half-hour with twelve values and D6 refits twice. Settings are
shrunk (1-week p-value windows, 3-day profiles, small forests) so a full pass takes about
a second; the real settings are the same code with larger numbers.
"""

from __future__ import annotations

import datetime as dt
from functools import cache

import numpy as np
import pandas as pd

from gridcast.anomalies.detectors import ROBUST_DISTANCE, Recipe
from gridcast.anomalies.run import Split
from gridcast.anomalies.series import MarketSeries, QuantileBand, epoch_minutes
from gridcast.config import AnomaliesConfig, load_config
from gridcast.timeutils import INTERVAL, NEM_TZ

TZ = "Australia/Melbourne"
START = dt.date(2024, 3, 20)  # NEM day of the first interval
N_DAYS = 45
N_LEADS = 288


def small_config() -> AnomaliesConfig:
    """The real anomalies config with windows shrunk for synthetic data."""
    a = load_config().anomalies
    return a.model_copy(
        update={
            "pvalues": a.pvalues.model_copy(update={"window_weeks": 1}),
            "profile": a.profile.model_copy(update={"days": 3}),
            "price_volatility": a.price_volatility.model_copy(update={"baseline_days": 1}),
            "isolation_forest": a.isolation_forest.model_copy(
                update={"n_estimators": 10, "max_samples": 64, "train_weeks": 1}
            ),
        }
    )


def small_v2_recipe() -> Recipe:
    """v2 (m4_plan.md §2.10) with the robust distance's window shrunk like the forest's."""
    a = small_config()
    v2 = load_config().anomalies_v2
    distance = v2.robust_distance.model_copy(update={"train_weeks": 1, "min_train_rows": 64})
    return Recipe(2, v2.price_level.scale_aud, a.price_level.scale_aud, ROBUST_DISTANCE, distance)


def split(cfg: AnomaliesConfig | None = None) -> Split:
    return Split(
        name="dev",
        market_first_day=START + dt.timedelta(days=9),
        residual_first_day=START + dt.timedelta(days=10),
        end_day=None,
    )


def grid(start: dt.date = START, n_days: int = N_DAYS) -> pd.DatetimeIndex:
    first = pd.Timestamp(start, tz=NEM_TZ).tz_convert("UTC") + INTERVAL
    return pd.date_range(first, periods=n_days * 288, freq="5min")


def synthetic_series(seed: int = 0, start: dt.date = START, n_days: int = N_DAYS) -> MarketSeries:
    rng = np.random.default_rng(seed)
    ends = grid(start, n_days)
    n = len(ends)
    local = (ends - INTERVAL).tz_convert(TZ)
    hour = np.asarray(local.hour + local.minute / 60.0)
    weekday = np.asarray(local.day_of_week) < 5
    daily = np.sin(2 * np.pi * (hour - 9) / 24)
    demand = 5000 + 900 * daily - 400 * ~weekday + rng.normal(0, 60, n)
    demand += np.cumsum(rng.normal(0, 8, n))  # slow drift
    jumps = rng.random(n) < 0.002
    demand[jumps] += rng.choice([-1, 1], jumps.sum()) * rng.uniform(800, 2500, jumps.sum())
    price = 70 + 60 * daily + rng.normal(0, 15, n)
    spikes = rng.random(n) < 0.004
    price[spikes] = rng.choice([450.0, 1200.0, 9000.0, -150.0], spikes.sum())
    solar_uigf = np.clip(2200 * np.sin(np.pi * (hour - 7) / 12), 0, None) * (hour > 7) * (hour < 19)
    wind_uigf = 900 + 500 * np.sin(np.arange(n) / 700.0) + rng.normal(0, 30, n)
    wind_cleared = wind_uigf * (1 - (rng.random(n) < 0.05) * rng.uniform(0, 0.5, n))
    solar_cleared = solar_uigf * (1 - (rng.random(n) < 0.08) * rng.uniform(0, 0.6, n))
    dispatchable = demand - wind_cleared - solar_cleared + rng.normal(0, 40, n)
    return MarketSeries(
        ends=ends,
        demand=demand,
        price=price,
        net_interchange=rng.normal(0, 300, n) + 200 * daily,
        available=demand + 2500 + rng.normal(0, 150, n),
        dispatchable=dispatchable,
        wind_cleared=wind_cleared,
        wind_uigf=wind_uigf,
        solar_cleared=solar_cleared,
        solar_uigf=solar_uigf,
        business=weekday,
        tz=TZ,
    )


def synthetic_band(series: MarketSeries, seed: int = 1, drop: float = 0.03) -> QuantileBand:
    """A 95% band around a noisy copy of demand, issued every 30 min, 288 leads."""
    rng = np.random.default_rng(seed)
    first = series.ends[0] - INTERVAL
    issues = pd.date_range(first.ceil("30min"), series.ends[-1], freq="30min")
    minutes = epoch_minutes(series.ends)
    m0 = minutes[0]
    leads = np.arange(1, N_LEADS + 1)
    target = epoch_minutes(issues)[:, None] + 5 * leads[None, :]
    idx = (target - m0) // 5
    inside = (idx >= 0) & (idx < len(series))
    centre = np.full(idx.shape, np.nan)
    centre[inside] = series.demand[idx[inside]] + rng.normal(0, 120, inside.sum())
    half = 1.96 * (100 + 1.2 * leads)[None, :]
    lower, upper = centre - half, centre + half
    valid = inside & (rng.random(idx.shape) >= drop)
    return QuantileBand(issues, lower, upper, valid)


@cache
def base_inputs() -> tuple[MarketSeries, QuantileBand]:
    series = synthetic_series()
    return series, synthetic_band(series)
