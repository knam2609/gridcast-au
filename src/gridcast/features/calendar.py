"""Calendar features: local-time, weekday, holiday, season and DST attributes.

All of these are known in advance for any future interval, so they are safe to use as
*target-time* features (no leakage). Local time is derived from ``interval_start`` in the
region's IANA zone (Australia/Melbourne observes daylight saving).
"""

from __future__ import annotations

import datetime as dt

import holidays
import numpy as np
import pandas as pd
from sqlalchemy import Connection, text

from gridcast.timeutils import local_day_n_intervals

CALENDAR_COLUMNS = [
    "jurisdiction",
    "local_date",
    "local_tz",
    "is_weekend",
    "is_public_holiday",
    "holiday_name",
    "is_dst_change_day",
    "n_intervals",
]

# Southern-hemisphere meteorological seasons.
_SEASON_BY_MONTH = {
    12: "summer", 1: "summer", 2: "summer",
    3: "autumn", 4: "autumn", 5: "autumn",
    6: "winter", 7: "winter", 8: "winter",
    9: "spring", 10: "spring", 11: "spring",
}  # fmt: skip


def build_calendar_days(
    jurisdiction: str, tz: str, start: dt.date, end: dt.date, country: str = "AU"
) -> pd.DataFrame:
    """One row per local calendar date in ``[start, end]`` for a holiday jurisdiction."""
    days = pd.date_range(start, end, freq="D").date
    hols = holidays.country_holidays(
        country, subdiv=jurisdiction, years=range(start.year, end.year + 1)
    )
    names = [hols.get(d) for d in days]
    n_intervals = [local_day_n_intervals(d, tz) for d in days]
    return pd.DataFrame(
        {
            "jurisdiction": jurisdiction,
            "local_date": days,
            "local_tz": tz,
            "is_weekend": [d.weekday() >= 5 for d in days],
            "is_public_holiday": [n is not None for n in names],
            "holiday_name": names,
            "is_dst_change_day": [n != 288 for n in n_intervals],
            "n_intervals": n_intervals,
        }
    )[CALENDAR_COLUMNS]


def upsert_calendar_days(conn: Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    records = [{str(k): v for k, v in r.items()} for r in df.to_dict(orient="records")]
    result = conn.execute(
        text(
            """
            INSERT INTO calendar_days
                (jurisdiction, local_date, local_tz, is_weekend, is_public_holiday,
                 holiday_name, is_dst_change_day, n_intervals)
            VALUES (:jurisdiction, :local_date, :local_tz, :is_weekend, :is_public_holiday,
                    :holiday_name, :is_dst_change_day, :n_intervals)
            ON CONFLICT (jurisdiction, local_date) DO UPDATE
               SET local_tz = EXCLUDED.local_tz,
                   is_weekend = EXCLUDED.is_weekend,
                   is_public_holiday = EXCLUDED.is_public_holiday,
                   holiday_name = EXCLUDED.holiday_name,
                   is_dst_change_day = EXCLUDED.is_dst_change_day,
                   n_intervals = EXCLUDED.n_intervals
            """
        ),
        records,
    )
    return int(result.rowcount)


def load_calendar_days(conn: Connection, jurisdiction: str) -> pd.DataFrame:
    df = pd.read_sql(
        text("SELECT * FROM calendar_days WHERE jurisdiction = :j ORDER BY local_date"),
        conn,
        params={"j": jurisdiction},
    )
    df["local_date"] = pd.to_datetime(df["local_date"]).dt.date
    return df


def interval_calendar_features(
    interval_start_utc: pd.Series, tz: str, calendar_days: pd.DataFrame
) -> pd.DataFrame:
    """Per-interval calendar features for UTC ``interval_start`` timestamps.

    Raises if any interval's local date is missing from ``calendar_days`` (so a stale
    calendar can never silently mark holidays as normal days).
    """
    local = interval_start_utc.dt.tz_convert(tz)
    local_date = local.dt.date
    cal = calendar_days.set_index("local_date")
    missing = ~local_date.isin(cal.index)
    if missing.any():
        raise KeyError(
            f"calendar_days missing {int(missing.sum())} interval dates, "
            f"e.g. {local_date[missing].iloc[0]}"
        )
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    is_holiday = local_date.map(cal["is_public_holiday"]).astype(bool)
    is_weekend = local.dt.day_of_week >= 5
    out = pd.DataFrame(
        {
            "local_start": local,
            "local_date": local_date,
            "local_hour": local.dt.hour.astype("int16"),
            "minute_of_day": minute_of_day.astype("int16"),
            "slot_of_day": (minute_of_day // 5).astype("int16"),
            "day_of_week": local.dt.day_of_week.astype("int8"),
            "month": local.dt.month.astype("int8"),
            "season": local.dt.month.map(_SEASON_BY_MONTH).astype("string"),
            "is_weekend": is_weekend.to_numpy(),
            "is_public_holiday": is_holiday.to_numpy(),
            "is_dst_change_day": local_date.map(cal["is_dst_change_day"]).astype(bool).to_numpy(),
        },
        index=interval_start_utc.index,
    )
    out["day_type"] = np.where(
        out["is_weekend"] | out["is_public_holiday"], "non_working", "working"
    )
    # Cyclical encodings (useful for linear models later).
    angle = 2 * np.pi * out["minute_of_day"] / 1440
    out["tod_sin"], out["tod_cos"] = np.sin(angle), np.cos(angle)
    return out
