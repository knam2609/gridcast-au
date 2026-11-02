"""M2 step 0: one month of AEMO pre-dispatch (30-min) and 5-minute pre-dispatch, VIC1.

usage: uv run python scripts/spikes/m2/aemo_forecasts.py CACHE_DIR [--month 2025-01]

Downloads PREDISPATCHREGIONSUM#ALL and P5MIN_REGIONSOLUTION#ALL for the month into
CACHE_DIR/nemweb (about 100 MB), then reports run identifiers, publication timing (LASTCHANGED),
horizon by run time, and how TOTALDEMAND relates to actual 5-minute demand from the project
database (start it with `make db-up`). Output feeds docs/m2_data_spike.md §4-5.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd

from gridcast.database.engine import get_engine
from gridcast.database.queries import load_market_frame

UA = "gridcast-au/0.1 (+https://github.com/knam2609/gridcast-au)"
ROOT = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
NEM = "Etc/GMT-10"  # NEM time: fixed UTC+10
REGION = "VIC1"


def download(cache: Path, month: pd.Period, folder: str, table: str) -> Path:
    y, m = month.year, f"{month.month:02d}"
    path = cache / f"{table}_ALL_{y}{m}.zip"
    if not path.exists():
        name = f"PUBLIC_ARCHIVE%23{table}%23ALL%23FILE01%23{y}{m}010000.zip"
        u = f"{ROOT}/{y}/MMSDM_{y}_{m}/MMSDM_Historical_Data_SQLLoader/{folder}/{name}"
        with httpx.stream("GET", u, headers={"User-Agent": UA}, timeout=300) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
            tmp.rename(path)
    return path


def read_region(path: Path, region_col: int = 6) -> tuple[list[str], pd.DataFrame]:
    """Stream the single CSV in an MMS zip; keep D rows for REGION, columns named by the I row."""
    with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as fh:
        header: list[str] | None = None
        rows = []
        for rec in csv.reader(io.TextIOWrapper(fh, encoding="ascii")):
            if rec[0] == "I":
                header = rec
            elif rec[0] == "D" and rec[region_col] == REGION:
                rows.append(rec)
    if header is None:
        raise ValueError(f"no I row in {path}")
    return header[:4], pd.DataFrame(rows, columns=header)


def nem_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y/%m/%d %H:%M:%S").dt.tz_localize(NEM)


def actual_demand() -> pd.Series:
    a = load_market_frame(get_engine(), REGION).set_index("interval_end")["total_demand_mw"]
    a.index = a.index.tz_convert(NEM)
    return a


def predispatch(path: Path, actual: pd.Series) -> None:
    ident, df = read_region(path)
    cols = ["PREDISPATCHSEQNO", "RUNNO", "PERIODID", "INTERVENTION", "TOTALDEMAND"]
    df = df[[*cols, "LASTCHANGED", "DATETIME"]].copy()
    df["LASTCHANGED"], df["DATETIME"] = nem_time(df.LASTCHANGED), nem_time(df.DATETIME)
    df["TOTALDEMAND"] = df.TOTALDEMAND.astype(float)
    print(f"=== {path.name}: {ident}, {REGION} rows {len(df):,}")
    print("INTERVENTION:", df.INTERVENTION.value_counts().to_dict())
    print("RUNNO:", df.RUNNO.value_counts().to_dict())
    key = ["PREDISPATCHSEQNO", "RUNNO", "INTERVENTION", "DATETIME"]
    print("duplicate keys", key, ":", int(df.duplicated(key).sum()))

    runs = df.groupby(["PREDISPATCHSEQNO", "RUNNO"]).agg(
        lc_min=("LASTCHANGED", "min"),
        lc=("LASTCHANGED", "max"),
        first_end=("DATETIME", "min"),
        last_end=("DATETIME", "max"),
        min_period=("PERIODID", "min"),
    )
    seq = runs.index.get_level_values(0)
    # Trading day starts 04:00 NEM time; run PP is nominally at 04:00 + (PP - 1) x 30 min.
    day = pd.to_datetime(seq.str[:8], format="%Y%m%d").tz_localize(NEM) + pd.Timedelta(hours=4)
    runs["nominal"] = day + (seq.str[8:].astype(int) - 1) * pd.Timedelta(minutes=30)
    n_days = seq.str[:8].nunique()
    print(f"runs: {len(runs):,} over {n_days} trading days ({len(runs) / n_days:.1f}/day)")
    print("LASTCHANGED identical within each run:", bool((runs.lc_min == runs.lc).all()))
    lag = (runs.lc - runs.nominal).dt.total_seconds()
    print("LASTCHANGED - nominal run time (s):", lag.describe().round(1).to_dict())
    first = (runs.first_end - runs.nominal).dt.total_seconds() / 60
    print("first period end - nominal (min):", first.value_counts().to_dict())
    print("first PERIODID of each run:", runs.min_period.value_counts().to_dict())
    runs["horizon_h"] = (runs.last_end - runs.lc).dt.total_seconds() / 3600
    hz = runs.groupby(runs.nominal.dt.strftime("%H:%M")).horizon_h.min().round(1)
    print("min horizon (last period end - LASTCHANGED, h) by nominal run time:")
    print("  " + ", ".join(f"{k} {v}" for k, v in hz.items()))
    print(f"runs reaching >= 24 h: {(runs.horizon_h >= 24).mean():.1%}")

    mean30 = actual.groupby(actual.index.ceil("30min")).mean()
    p = df[df.INTERVENTION == "0"].join(runs.lc.rename("run_lc"), on=["PREDISPATCHSEQNO", "RUNNO"])
    p["actual_mean30"] = mean30.reindex(p.DATETIME).to_numpy()
    p["actual_at_end"] = actual.reindex(p.DATETIME).to_numpy()
    before = actual.reindex(p.DATETIME - pd.Timedelta(minutes=30)).to_numpy()
    p["ramp30"] = p.actual_at_end - before
    p = p.dropna(subset=["actual_mean30", "actual_at_end"])
    p["lead_h"] = (p.DATETIME - p.run_lc).dt.total_seconds() / 3600
    bands = pd.cut(p.lead_h, [0, 0.5, 1, 2, 6, 12, 24, 40])
    for name, g in [("all periods", p), ("|30-min ramp| >= 300 MW", p[p.ramp30.abs() >= 300])]:
        e_mean = g.TOTALDEMAND - g.actual_mean30
        e_end = g.TOTALDEMAND - g.actual_at_end
        gb = g.lead_h.groupby(bands.loc[g.index], observed=True)
        out = pd.DataFrame(
            {
                "n": gb.size(),
                "MAE_vs_30min_mean": e_mean.abs().groupby(bands.loc[g.index], observed=True).mean(),
                "MAE_vs_period_end": e_end.abs().groupby(bands.loc[g.index], observed=True).mean(),
                "bias_vs_30min_mean": e_mean.groupby(bands.loc[g.index], observed=True).mean(),
                "bias_vs_period_end": e_end.groupby(bands.loc[g.index], observed=True).mean(),
            }
        )
        print(f"-- TOTALDEMAND vs actual (MW), {name}, by lead from LASTCHANGED (h)")
        print(out.round(1).to_string())


def p5min(path: Path, actual: pd.Series) -> None:
    ident, df = read_region(path)
    df = df[["RUN_DATETIME", "INTERVAL_DATETIME", "INTERVENTION", "TOTALDEMAND", "LASTCHANGED"]]
    df = df.assign(
        RUN_DATETIME=nem_time(df.RUN_DATETIME),
        INTERVAL_DATETIME=nem_time(df.INTERVAL_DATETIME),
        LASTCHANGED=nem_time(df.LASTCHANGED),
        TOTALDEMAND=df.TOTALDEMAND.astype(float),
    )
    print(f"\n=== {path.name}: {ident}, {REGION} rows {len(df):,}")
    print("INTERVENTION:", df.INTERVENTION.value_counts().to_dict())
    g = df.groupby("RUN_DATETIME").agg(
        n=("INTERVAL_DATETIME", "nunique"),
        first=("INTERVAL_DATETIME", "min"),
        last=("INTERVAL_DATETIME", "max"),
        lc=("LASTCHANGED", "max"),
    )
    print(f"runs: {len(g):,}; intervals per run: {g.n.value_counts().to_dict()}")
    to_min = pd.Timedelta("1min")
    print("first interval end - RUN_DATETIME (min):", ((g["first"] - g.index) / to_min).unique())
    print("last interval end - RUN_DATETIME (min):", ((g["last"] - g.index) / to_min).unique())
    early = (g.index - g.lc).dt.total_seconds()
    print("RUN_DATETIME - LASTCHANGED (s):", early.describe().round(1).to_dict())
    df = df.assign(
        actual=actual.reindex(df.INTERVAL_DATETIME).to_numpy(),
        step=((df.INTERVAL_DATETIME - df.RUN_DATETIME) / pd.Timedelta("5min")).astype(int),
    ).dropna(subset=["actual"])
    mae = (df.TOTALDEMAND - df.actual).abs().groupby(df.step).mean().round(1)
    print("MAE vs actual 5-min demand (MW) by interval index in run (0 = RUN_DATETIME):")
    print("  " + ", ".join(f"{k}: {v}" for k, v in mae.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--month", default="2025-01")
    args = ap.parse_args()
    cache = args.cache_dir / "nemweb"
    cache.mkdir(parents=True, exist_ok=True)
    month = pd.Period(args.month, freq="M")
    actual = actual_demand()
    pd.set_option("display.width", 250)
    predispatch(download(cache, month, "PREDISP_ALL_DATA", "PREDISPATCHREGIONSUM"), actual)
    p5min(download(cache, month, "P5MIN_ALL_DATA", "P5MIN_REGIONSOLUTION"), actual)


if __name__ == "__main__":
    main()
