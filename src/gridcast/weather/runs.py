"""When an archived forecast value was produced and when it became usable.

Open-Meteo's Previous Runs API returns, for valid hour *t* and offset *K* days
(``previous_dayK``), the value from the run initialised at ``floor_cadence(t − 24·K h)``.
Step 0 verified this exactly for GFS (6-hourly runs, docs/m2_data_spike.md §3). A value is
usable at ``run_init + publication_delay``; nothing in the archive records the real
publication time, so the delay is a configured, conservative constant.
"""

from __future__ import annotations

import pandas as pd

HOUR = pd.Timedelta(hours=1)


def _require_utc(values: pd.Series) -> None:
    if not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dtype.tz) != "UTC":
        raise ValueError("expected tz-aware UTC timestamps")


def run_init_for(valid_time: pd.Series, offset_days: int, run_every: pd.Timedelta) -> pd.Series:
    """Initialisation time of the run that supplies ``previous_day{offset_days}``.

    Uses fixed 24-hour days (never calendar days), and floors relative to 00:00 UTC.
    """
    _require_utc(valid_time)
    if offset_days < 1:
        raise ValueError("offset_days must be >= 1")
    return (valid_time - pd.Timedelta(hours=24 * offset_days)).dt.floor(run_every)


def available_at_for(run_init: pd.Series, publication_delay: pd.Timedelta) -> pd.Series:
    _require_utc(run_init)
    return run_init + publication_delay
