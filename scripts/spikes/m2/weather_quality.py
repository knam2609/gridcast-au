"""M2 step 0: are archived forecasts genuine, where are the gaps, and how skilful are they vs ERA5?

usage: uv run python scripts/spikes/m2/weather_quality.py CACHE_DIR

Needs the Previous Runs JSON cached by weather_coverage.py. Downloads ERA5 (Historical Weather
API) for Melbourne and GFS temperature for a second site (Latrobe Valley) to confirm the GFS gap is
archive-wide. Output feeds docs/m2_data_spike.md §1-2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import pandas as pd

UA = "gridcast-au/0.1 (+https://github.com/knam2609/gridcast-au)"
WINDOW = ("2021-10-01", "2026-09-30 23:00")  # the M1/M2 study window, UTC hours
VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
]
MODELS = ["gfs_seamless", "jma_seamless", "ecmwf_ifs025", "icon_seamless"]


def get_json(path: Path, url: str, params: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    r = httpx.get(url, params=params, headers={"User-Agent": UA}, timeout=180)
    r.raise_for_status()
    path.write_text(r.text)
    time.sleep(1.0)
    return r.json()


def frame(d: dict) -> pd.DataFrame:
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h.set_index("time").loc[WINDOW[0] : WINDOW[1]]


def gap_runs(s: pd.Series) -> pd.DataFrame:
    na = s.isna()
    if not na.any():
        return pd.DataFrame(columns=["start", "end", "hours"])
    run_id = (na != na.shift()).cumsum()[na]
    g = s[na].groupby(run_id)
    return pd.DataFrame(
        {
            "start": g.apply(lambda x: x.index[0]),
            "end": g.apply(lambda x: x.index[-1]),
            "hours": g.size(),
        }
    ).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    cache = ap.parse_args().cache_dir / "open_meteo"
    pd.set_option("display.width", 250)

    def load(model: str, var: str) -> pd.DataFrame:
        return frame(json.loads((cache / f"{model}__{var}.json").read_text()))

    # --- 1. Genuineness and gaps (study window only)
    for model, var in [
        ("gfs_seamless", "temperature_2m"),
        ("jma_seamless", "temperature_2m"),
        ("jma_seamless", "cloud_cover"),
        ("jma_seamless", "relative_humidity_2m"),
    ]:
        h = load(model, var)
        d0, d1, d2 = h[var], h[f"{var}_previous_day1"], h[f"{var}_previous_day2"]
        print(f"\n=== {model} {var} ({len(h)} hours in window)")
        for name, s in [("day1", d1), ("day2", d2)]:
            g = gap_runs(s)
            print(
                f"  {name}: missing {int(s.isna().sum())} h ({100 * s.isna().mean():.2f}%); "
                f"gap runs: {g.to_dict('records')}"
            )
        yr = h.index.year
        print(
            pd.DataFrame(
                {
                    "d1_equals_d0_pct": (d1 == d0).groupby(yr).mean() * 100,
                    "MAE_d1_vs_d0": (d1 - d0).abs().groupby(yr).mean(),
                    "MAE_d2_vs_d0": (d2 - d0).abs().groupby(yr).mean(),
                }
            )
            .round(3)
            .to_string()
        )

    latrobe = get_json(
        cache / "latrobe__gfs_seamless__temperature_2m.json",
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        {
            "latitude": -38.23,
            "longitude": 146.40,
            "models": "gfs_seamless",
            "timezone": "GMT",
            "start_date": "2021-10-01",
            "end_date": "2026-09-30",
            "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
        },
    )
    lt = frame(latrobe)
    print("\n=== Latrobe Valley (-38.23, 146.40) gfs_seamless temperature gaps")
    for c in lt.columns:
        print(f"  {c}: {gap_runs(lt[c]).to_dict('records')}")

    # --- 2. Skill against ERA5
    era5 = frame(
        get_json(
            cache / "era5.json",
            "https://archive-api.open-meteo.com/v1/archive",
            {
                "latitude": -37.83,
                "longitude": 144.98,
                "models": "era5",
                "timezone": "GMT",
                "start_date": "2021-10-01",
                "end_date": "2026-09-30",
                "hourly": ",".join([*VARS, "wind_speed_10m"]),
            },
        )
    )
    print(f"\nERA5 missing values in window: {int(era5.isna().sum().sum())}")
    for var in VARS:
        rows = {}
        for model in MODELS:
            f = load(model, var)
            for k in (1, 2):
                err = (f[f"{var}_previous_day{k}"] - era5[var]).dropna()
                if err.empty:
                    continue
                by_year = err.abs().groupby(err.index.year).mean()
                by_year.index = by_year.index.astype(str)
                by_year["bias"] = err.mean()
                by_year["hours"] = len(err)
                rows[f"{model} d{k}"] = by_year
        print(f"\n=== {var}: MAE vs ERA5 by year (hours where the forecast exists)")
        print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
