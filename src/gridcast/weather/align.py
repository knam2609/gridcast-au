"""Hourly weather → population-weighted, 5-minute values aligned with market intervals.

Rules (m2_plan.md §2.1, decided in step 0):

* Targets are 5-minute ``interval_end`` timestamps (UTC). Instant variables are interpolated
  linearly between the hours ``floor_1h(t)`` and ``ceil_1h(t)``; on the hour, the hour's own
  value is used.
* ``preceding_hour_mean`` variables (shortwave radiation) are means over the hour ending at
  the label, so the value labelled ``ceil_1h(t)`` applies to every interval ending in
  ``(ceil_1h(t) − 1 h, ceil_1h(t)]``.
* A 5-minute value exists only if every hourly value it uses exists, and it becomes
  available only when the **later** of those hours is available
  (``available_at = max(available_at[floor], available_at[ceil])``).
* A site-weighted value exists only if every weighted site has a value (no renormalising
  over the sites that happen to be present).

These functions are pure: they never look at issue times. Selecting, for an issue time *o*,
the freshest offset with ``available_at <= o`` is the feature builder's job (step 2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)
AVAILABLE_AT = "available_at"


def _check_utc(values: pd.Series | pd.DatetimeIndex, what: str) -> None:
    dtype = values.dtype
    if not isinstance(dtype, pd.DatetimeTZDtype) or str(dtype.tz) != "UTC":
        raise ValueError(f"{what} must be tz-aware UTC")


def site_weighted(
    hourly: pd.DataFrame,
    weights: Mapping[str, float],
    variables: Sequence[str],
    by: Sequence[str] = (),
) -> pd.DataFrame:
    """Population-weighted mean over sites for each ``(*by, valid_time)``.

    ``hourly`` has columns ``site_id``, ``valid_time``, ``*by``, ``*variables`` and optionally
    ``available_at`` (the latest of the sites' values is kept). Returns one row per key.
    """
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"site weights must sum to 1, got {sum(weights.values())}")
    _check_utc(hourly["valid_time"], "valid_time")
    keys = [*by, "valid_time"]
    if hourly.duplicated([*keys, "site_id"]).any():
        raise ValueError("duplicate (key, site_id) rows in hourly weather")
    sites = list(weights)
    w = pd.Series(weights, dtype="float64")
    wide = hourly[hourly["site_id"].isin(sites)].set_index([*keys, "site_id"])
    columns: dict[str, pd.Series] = {}
    # unstack, not pivot_table: pivot_table would silently aggregate and drop missing values.
    for var in variables:
        per_site = wide[var].astype("float64").unstack("site_id").reindex(columns=sites)  # noqa: PD010
        columns[var] = per_site.mul(w, axis=1).sum(axis=1, min_count=len(sites))
    if AVAILABLE_AT in hourly.columns:
        per_site_av = wide[AVAILABLE_AT].unstack("site_id").reindex(columns=sites)  # noqa: PD010
        columns[AVAILABLE_AT] = per_site_av.max(axis=1)
    out = pd.DataFrame(columns).sort_index()
    return out.reset_index()


def _to_five_minute_one(
    hourly: pd.DataFrame, targets: pd.DatetimeIndex, kinds: Mapping[str, str]
) -> pd.DataFrame:
    if hourly["valid_time"].duplicated().any():
        raise ValueError("duplicate valid_time in hourly weather")
    series = hourly.set_index("valid_time")
    hours = pd.DatetimeIndex(series.index)
    if not (hours.floor(HOUR) == hours).all():
        raise ValueError("hourly weather must be on whole hours")
    lo = targets.floor(HOUR)
    hi = targets.ceil(HOUR)
    frac = ((targets - lo) / HOUR).to_numpy(dtype="float64")
    out = pd.DataFrame({"interval_end": targets})
    for var, kind in kinds.items():
        values = series[var].astype("float64")
        at_hi = values.reindex(hi).to_numpy()
        if kind == "preceding_hour_mean":
            out[var] = at_hi
        elif kind == "instant":
            at_lo = values.reindex(lo).to_numpy()
            # On the hour lo == hi and frac == 0, so only that hour's value is used.
            out[var] = np.where(frac == 0.0, at_lo, (1.0 - frac) * at_lo + frac * at_hi)
        else:
            raise ValueError(f"unknown weather variable kind {kind!r} for {var}")
    if AVAILABLE_AT in series.columns:
        av = series[AVAILABLE_AT]
        both = pd.DataFrame(
            {"lo": av.reindex(lo).to_numpy(), "hi": av.reindex(hi).to_numpy()}, index=out.index
        )
        # Missing either bracketing hour => never available (NaT), not "available early".
        out[AVAILABLE_AT] = both.max(axis=1, skipna=False)
    return out


def to_five_minute(
    hourly: pd.DataFrame,
    targets: pd.DatetimeIndex,
    kinds: Mapping[str, str],
    by: str | None = None,
) -> pd.DataFrame:
    """Align hourly weather to 5-minute ``interval_end`` targets.

    ``kinds`` maps each variable to ``"instant"`` or ``"preceding_hour_mean"``. With ``by``
    (e.g. ``"offset_days"``), each group is aligned separately and the key is kept.
    """
    _check_utc(targets, "targets")
    _check_utc(hourly["valid_time"], "valid_time")
    if by is None:
        return _to_five_minute_one(hourly, targets, kinds)
    parts = []
    for key, group in hourly.groupby(by, sort=True):
        part = _to_five_minute_one(group.drop(columns=by), targets, kinds)
        part.insert(0, by, key)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)
