"""M2 step 0: which model run supplies previous_day1/2, how late is it published, and how often
can day1 be used?

usage: uv run python scripts/spikes/m2/weather_run_timing.py CACHE_DIR

1. Matches archived Previous Runs values to individual runs from the Single Runs API
   (6-hourly runs, 2026-09-24 → 2026-10-05; that archive only keeps recent months).
2. Prints Open-Meteo's metadata for the latest run (a point-in-time sample of publication delay).
3. Computes the share of (issue time, lead) pairs that may use previous_day1 under the exact rule
   (run init + delay) and under the original plan's simple bound.
Output feeds docs/m2_data_spike.md §3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import httpx
import pandas as pd

UA = "gridcast-au/0.1 (+https://github.com/knam2609/gridcast-au)"
LAT, LON = -37.83, 144.98
VARS = ["temperature_2m", "cloud_cover"]
# single-runs model id -> previous-runs model id (Single Runs maps gfs_seamless to HRRR).
MODELS = {"gfs_global": "gfs_seamless", "jma_gsm": "jma_seamless"}
RUNS = pd.date_range("2026-09-24T00:00", "2026-10-05T18:00", freq="6h")
VALID = pd.date_range("2026-09-28T00:00", "2026-10-04T23:00", freq="1h")
DELAY = pd.Timedelta(hours=6)


def get_json(path: Path, url: str, params: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    r = httpx.get(url, params=params, headers={"User-Agent": UA}, timeout=120)
    path.write_text(r.text)
    time.sleep(0.6)
    return r.json()


def hourly(d: dict) -> pd.DataFrame:
    f = pd.DataFrame(d["hourly"])
    f["time"] = pd.to_datetime(f["time"])
    return f.set_index("time")


def match_runs(cache: Path) -> None:
    for sr_id, pr_id in MODELS.items():
        prev = hourly(
            get_json(
                cache / f"prev__{pr_id}.json",
                "https://previous-runs-api.open-meteo.com/v1/forecast",
                {
                    "latitude": LAT,
                    "longitude": LON,
                    "models": pr_id,
                    "timezone": "GMT",
                    "start_date": "2026-09-28",
                    "end_date": "2026-10-04",
                    "hourly": ",".join(f"{v}_previous_day{k}" for v in VARS for k in (1, 2)),
                },
            )
        )
        runs = {}
        for run in RUNS:
            d = get_json(
                cache / f"{sr_id}__{run:%Y%m%d%H}.json",
                "https://single-runs-api.open-meteo.com/v1/forecast",
                {
                    "latitude": LAT,
                    "longitude": LON,
                    "models": sr_id,
                    "timezone": "GMT",
                    "run": run.strftime("%Y-%m-%dT%H:%M"),
                    "forecast_hours": 120,
                    "hourly": ",".join(VARS),
                },
            )
            if not d.get("error"):
                runs[run] = hourly(d)
        print(f"\n=== {sr_id} vs previous-runs {pr_id}: {len(runs)}/{len(RUNS)} runs fetched")
        for k in (1, 2):
            cols = [f"{v}_previous_day{k}" for v in VARS]
            recs = []
            for t in VALID:
                target = prev.loc[t, cols].to_numpy()
                exact = [
                    r
                    for r, f in runs.items()
                    if r <= t and t in f.index and (f.loc[t, VARS].to_numpy() == target).all()
                ]
                rule = (t - pd.Timedelta(hours=24 * k)).floor("6h")
                rule_err = abs(runs[rule].loc[t, "temperature_2m"] - target[0])
                recs.append(
                    {
                        "hour": t.hour,
                        "n_exact": len(exact),
                        "rule_matches": rule in exact,
                        "rule_abs_err_t2m": rule_err,
                    }
                )
            df = pd.DataFrame(recs)
            print(
                f"-- previous_day{k}: hours where run floor6h(t-{24 * k}h) matches exactly: "
                f"{df.rule_matches.mean():.1%}; hours with no exact match at all: "
                f"{(df.n_exact == 0).mean():.1%}; |temp error| of rule run, by UTC hour: "
                f"{df.groupby('hour').rule_abs_err_t2m.mean().round(2).to_dict()}"
            )


def publication_delay() -> None:
    now = dt.datetime.now(dt.UTC)
    print(f"\n=== Open-Meteo latest-run metadata (sampled {now:%Y-%m-%d %H:%M}Z)")
    for m in ["ncep_gfs013", "ncep_gfs025", "jma_gsm"]:
        url = f"https://api.open-meteo.com/data/{m}/static/meta.json"
        d = httpx.get(url, headers={"User-Agent": UA}, timeout=60).json()
        init, avail = d["last_run_initialisation_time"], d["last_run_availability_time"]

        def iso(x: int) -> str:
            return dt.datetime.fromtimestamp(x, dt.UTC).strftime("%Y-%m-%d %H:%M")

        delay_h = (avail - init) / 3600
        print(
            f"  {m}: init {iso(init)}Z, available {iso(avail)}Z, delay {delay_h:.2f} h,"
            f" native step {d['temporal_resolution_seconds'] // 3600} h"
        )


def offset_usage() -> None:
    """A 5-minute target interpolates between two hours; the later, ceil_1h(t), must be known."""
    rows = []
    for o in pd.date_range("2025-01-01", periods=48, freq="30min"):  # covers every 6-h phase
        for lead in pd.to_timedelta(range(5, 1441, 5), unit="min"):
            t = (o + lead).ceil("1h")
            rows.append(
                {
                    "lead_h": lead / pd.Timedelta("1h"),
                    "exact_day1": (t - pd.Timedelta(hours=24)).floor("6h") + DELAY <= o,
                    "simple_day1": t - pd.Timedelta(hours=24) + DELAY <= o,
                    "exact_day2": (t - pd.Timedelta(hours=48)).floor("6h") + DELAY <= o,
                }
            )
    df = pd.DataFrame(rows)
    bands = pd.cut(df.lead_h, [0, 12, 18, 20, 22, 23, 24])
    print("\n=== share of (issue, lead) pairs where the offset is available (%), delay 6 h")
    cols = ["exact_day1", "simple_day1", "exact_day2"]
    print(df.groupby(bands, observed=True)[cols].mean().mul(100).round(1).to_string())
    print(
        f"overall day1: exact {100 * df.exact_day1.mean():.1f}%, "
        f"simple {100 * df.simple_day1.mean():.1f}%; day2 always available: {df.exact_day2.all()}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    cache = ap.parse_args().cache_dir / "open_meteo" / "single_runs"
    cache.mkdir(parents=True, exist_ok=True)
    match_runs(cache)
    publication_delay()
    offset_usage()


if __name__ == "__main__":
    main()
