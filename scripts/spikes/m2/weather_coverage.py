"""M2 step 0: Open-Meteo Previous Runs coverage per model x variable x offset (Melbourne).

usage: uv run python scripts/spikes/m2/weather_coverage.py CACHE_DIR [--models a,b,c]

Downloads (and caches as JSON under CACHE_DIR/open_meteo) the full 2021-01 → 2026-10 hourly series
for each model and variable, with offsets previous_day0..2, then prints the first valid hour and
the share of non-missing hours after it. Output feeds docs/m2_data_spike.md §1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import pandas as pd

UA = "gridcast-au/0.1 (+https://github.com/knam2609/gridcast-au)"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = -37.83, 144.98  # Melbourne
VARS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
]
OFFSETS = [0, 1, 2]
START, END = "2021-01-01", "2026-10-31"
DEFAULT_MODELS = (
    "gfs_seamless,gfs_global,ecmwf_ifs025,bom_access_global,icon_seamless,jma_seamless,gem_seamless"
)


def fetch(cache: Path, model: str, var: str) -> dict:
    path = cache / f"{model}__{var}.json"
    if path.exists():
        return json.loads(path.read_text())
    cols = [var if k == 0 else f"{var}_previous_day{k}" for k in OFFSETS]
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(cols),
        "models": model,
        "start_date": START,
        "end_date": END,
        "timezone": "GMT",
    }
    r = httpx.get(URL, params=params, headers={"User-Agent": UA}, timeout=120)
    data = r.json() if r.status_code == 200 else {"error": True, "status": r.status_code}
    path.write_text(json.dumps(data))
    time.sleep(1.0)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    args = ap.parse_args()
    cache = args.cache_dir / "open_meteo"
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in args.models.split(","):
        for var in VARS:
            d = fetch(cache, model, var)
            if d.get("error"):
                print(f"{model} {var}: ERROR {d}")
                continue
            h = pd.DataFrame(d["hourly"])
            h["time"] = pd.to_datetime(h["time"])
            h = h.set_index("time")
            for c in h.columns:
                s = h[c]
                ok = s.notna()
                first = s.index[ok.argmax()] if ok.any() else None
                after = s[first:] if first is not None else s.iloc[0:0]
                rows.append(
                    {
                        "model": model,
                        "column": c,
                        "grid_point": f"({d['latitude']:.2f}, {d['longitude']:.2f})",
                        "first_valid": first,
                        "non_null_after_first_pct": round(100 * after.notna().mean(), 2)
                        if len(after)
                        else 0.0,
                        "missing_hours_after_first": int(after.isna().sum()),
                        "last_valid": s.index[::-1][ok[::-1].argmax()] if ok.any() else None,
                    }
                )
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
