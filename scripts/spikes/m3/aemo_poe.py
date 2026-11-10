"""M3 step 0: AEMO's probabilistic (POE) demand forecasts in PDPASA_REGIONSOLUTION, VIC1.

usage: uv run python scripts/spikes/m3/aemo_poe.py CACHE_DIR [--month 2025-01]

Downloads the month's MMSDM `DATA/PDPASA_REGIONSOLUTION` file into CACHE_DIR/nemweb (about
70 MB zipped, 260 MB unzipped) and reports, for VIC1:
  1. structure: table version, RUNTYPE values, runs per day, duplicate keys, reach per run;
  2. timing: LASTCHANGED against RUN_DATETIME, and which pre-dispatch run each PDPASA run
     matches (the nominal-run offset at which DEMAND50 is closest to pre-dispatch TOTALDEMAND);
  3. the POE band: DEMAND10 - DEMAND50 and DEMAND50 - DEMAND90, in MW and as % of DEMAND50;
  4. DEMAND50 against actual demand (30-minute mean of 5-minute TOTALDEMAND);
  5. how often actual demand falls below DEMAND90, inside [DEMAND90, DEMAND10], above DEMAND10.
Actual demand and pre-dispatch come from the project database (start it with `make db-up`).
Output feeds docs/m3_data_spike.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd
from sqlalchemy import text

from gridcast.database.engine import get_engine
from gridcast.database.queries import load_market_frame

UA = "gridcast-au/0.1 (+https://github.com/knam2609/gridcast-au)"
ROOT = "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
NEM = "Etc/GMT-10"  # NEM time: fixed UTC+10
REGION = "VIC1"
TABLE = "PDPASA_REGIONSOLUTION"
HALF_HOUR = pd.Timedelta(minutes=30)


def download(cache: Path, month: pd.Period) -> Path:
    y, m = month.year, f"{month.month:02d}"
    path = cache / f"{TABLE}_{y}{m}.zip"
    if not path.exists():
        name = f"PUBLIC_ARCHIVE%23{TABLE}%23FILE01%23{y}{m}010000.zip"
        u = f"{ROOT}/{y}/MMSDM_{y}_{m}/MMSDM_Historical_Data_SQLLoader/DATA/{name}"
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


def actual_mean30() -> pd.Series:
    """Actual demand averaged over each 30-minute period, indexed by period end (NEM time)."""
    a = load_market_frame(get_engine(), REGION).set_index("interval_end")["total_demand_mw"]
    a.index = a.index.tz_convert(NEM)
    counts = a.groupby(a.index.ceil("30min")).count()
    mean = a.groupby(a.index.ceil("30min")).mean()
    return mean[counts == 6]


def predispatch(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    q = text(
        "SELECT run_time, target_end, total_demand_mw FROM aemo_demand_forecasts "
        "WHERE source = 'predispatch' AND region_id = :r AND run_time >= :a AND run_time < :b"
    )
    with get_engine().connect() as c:
        df = pd.read_sql(q, c, params={"r": REGION, "a": start, "b": end})
    df["run_time"] = df.run_time.dt.tz_convert(NEM)
    df["target_end"] = df.target_end.dt.tz_convert(NEM)
    return df


def structure(df: pd.DataFrame, ident: list[str]) -> pd.DataFrame:
    print(f"=== {TABLE}: {ident}, {REGION} rows {len(df):,}")
    print("RUNTYPE:", df.RUNTYPE.value_counts().to_dict())
    key = ["RUN_DATETIME", "INTERVAL_DATETIME", "RUNTYPE"]
    print("duplicate keys", key, ":", int(df.duplicated(key).sum()))
    runs = df.groupby(["RUN_DATETIME", "RUNTYPE"]).agg(
        n=("INTERVAL_DATETIME", "nunique"),
        first=("INTERVAL_DATETIME", "min"),
        last=("INTERVAL_DATETIME", "max"),
        lc_min=("LASTCHANGED", "min"),
        lc=("LASTCHANGED", "max"),
    )
    for rt, g in runs.groupby(level="RUNTYPE"):
        t = g.index.get_level_values("RUN_DATETIME")
        days = t.normalize().nunique()
        step = pd.Series(t).diff().dropna().value_counts().head(3)
        print(f"-- RUNTYPE {rt}: {len(g):,} runs over {days} days ({len(g) / days:.1f}/day)")
        print("   spacing between runs:", {str(k): int(v) for k, v in step.items()})
        print("   LASTCHANGED identical within each run:", bool((g.lc_min == g.lc).all()))
        first = (g["first"] - t) / pd.Timedelta("1min")
        print("   first INTERVAL_DATETIME - RUN_DATETIME (min):", first.value_counts().to_dict())
        lag = (g.lc - t).dt.total_seconds()
        print("   LASTCHANGED - RUN_DATETIME (s):", lag.describe().round(0).to_dict())
        reach = (g["last"] - g.lc).dt.total_seconds() / 3600
        print("   reach, last period end - LASTCHANGED (h):", reach.describe().round(1).to_dict())
        print(f"   runs reaching >= 24 h after publication: {(reach >= 24).mean():.1%}")
    return runs


def run_matching(df: pd.DataFrame, pd_fc: pd.DataFrame) -> None:
    """Which nominal pre-dispatch run does each PDPASA run match? Try offsets of RUN_DATETIME."""
    print("-- DEMAND50 vs pre-dispatch TOTALDEMAND (same period end), by assumed nominal run time")
    pre = pd_fc.set_index(["run_time", "target_end"]).total_demand_mw
    for off in (-60, -30, 0, 30):
        key = pd.MultiIndex.from_arrays(
            [df.RUN_DATETIME + pd.Timedelta(minutes=off), df.INTERVAL_DATETIME]
        )
        p = pre.reindex(key).to_numpy()
        d = df.DEMAND50.to_numpy() - p
        ok = ~pd.isna(p)
        diff = pd.Series(d[ok])
        print(
            f"   nominal = RUN_DATETIME {off:+d} min: matched {ok.mean():.1%}, "
            f"|diff| median {diff.abs().median():.1f}, mean {diff.abs().mean():.1f}, "
            f"exact (<0.5 MW) {(diff.abs() < 0.5).mean():.1%}"
        )


def definition(df: pd.DataFrame, pd_fc: pd.DataFrame) -> None:
    """Is DEMAND50 on the TOTALDEMAND definition? Compare with the matched pre-dispatch run
    (nominal = RUN_DATETIME - 30 min) and with actual demand on the same rows."""
    pre = pd_fc.set_index(["run_time", "target_end"]).total_demand_mw
    key = pd.MultiIndex.from_arrays([df.RUN_DATETIME - HALF_HOUR, df.INTERVAL_DATETIME])
    g = df.assign(pre=pre.reindex(key).to_numpy()).dropna(subset=["pre", "actual"])
    d = g.DEMAND50 - g.pre
    stats = d.describe().round(1).to_dict()
    print("-- DEMAND50 - pre-dispatch TOTALDEMAND (MW), matched run:", stats)
    print(f"   share negative: {(d < 0).mean():.2%}; share below -10 MW: {(d < -10).mean():.2%}")
    by_lead = d.groupby(g.lead_band, observed=True).agg(["mean", "median", "std"])
    print("   by lead (h):")
    print(by_lead.round(1).to_string())
    hour = g.INTERVAL_DATETIME.dt.hour
    print("   mean by NEM hour of the period end:")
    print("  ", d.groupby(hour).mean().round(0).astype(int).to_dict())
    e50, epre = g.DEMAND50 - g.actual, g.pre - g.actual
    out = pd.DataFrame(
        {
            "n": e50.groupby(g.lead_band, observed=True).size(),
            "MAE_D50": e50.abs().groupby(g.lead_band, observed=True).mean(),
            "MAE_pre": epre.abs().groupby(g.lead_band, observed=True).mean(),
            "bias_D50": e50.groupby(g.lead_band, observed=True).mean(),
            "bias_pre": epre.groupby(g.lead_band, observed=True).mean(),
        }
    )
    print("   DEMAND50 and pre-dispatch TOTALDEMAND vs actual 30-minute mean, same rows:")
    print(out.round(1).to_string())


def band(df: pd.DataFrame) -> None:
    up = df.DEMAND10 - df.DEMAND50
    down = df.DEMAND50 - df.DEMAND90
    print("-- POE band (MW): DEMAND10 - DEMAND50", up.describe().round(1).to_dict())
    print("                  DEMAND50 - DEMAND90", down.describe().round(1).to_dict())
    ordered = ((up >= 0) & (down >= 0)).mean()
    print(f"   share with DEMAND10 >= DEMAND50 >= DEMAND90: {ordered:.1%}")
    asym = (up - down).abs().max()
    print(f"   max |(DEMAND10 - DEMAND50) - (DEMAND50 - DEMAND90)|: {asym:.2f} MW")
    rel_up, rel_down = 100 * up / df.DEMAND50, 100 * down / df.DEMAND50
    print("   as % of DEMAND50: up", rel_up.describe().round(2).to_dict())
    print("                     down", rel_down.describe().round(2).to_dict())
    lead = df.lead_h.groupby(df.lead_band, observed=True)
    out = pd.DataFrame(
        {
            "n": lead.size(),
            "up_MW_mean": up.groupby(df.lead_band, observed=True).mean(),
            "down_MW_mean": down.groupby(df.lead_band, observed=True).mean(),
            "up_pct_mean": rel_up.groupby(df.lead_band, observed=True).mean(),
            "up_pct_std": rel_up.groupby(df.lead_band, observed=True).std(),
        }
    )
    print("   by lead from LASTCHANGED (h):")
    print(out.round(2).to_string())


def against_actual(df: pd.DataFrame) -> None:
    g = df.dropna(subset=["actual"])
    e = g.DEMAND50 - g.actual
    below, above = g.actual < g.DEMAND90, g.actual > g.DEMAND10
    inside = ~below & ~above
    by = g.lead_band
    width = g.DEMAND10 - g.DEMAND90
    lo, hi = g.DEMAND90 - g.actual, g.actual - g.DEMAND10
    alpha = 0.2
    score = width + (2 / alpha) * lo.clip(lower=0) + (2 / alpha) * hi.clip(lower=0)
    out = pd.DataFrame(
        {
            "n": e.groupby(by, observed=True).size(),
            "MAE_D50": e.abs().groupby(by, observed=True).mean(),
            "bias_D50": e.groupby(by, observed=True).mean(),
            "below_D90": below.groupby(by, observed=True).mean(),
            "inside": inside.groupby(by, observed=True).mean(),
            "above_D10": above.groupby(by, observed=True).mean(),
            "width": width.groupby(by, observed=True).mean(),
            "IS80_if_80pct": score.groupby(by, observed=True).mean(),
        }
    )
    print("-- DEMAND50 and the POE band vs actual 30-minute mean demand, by lead (h)")
    print(out.round(3).to_string())
    print(
        f"   pooled: n {len(g):,}, MAE {e.abs().mean():.1f}, bias {e.mean():.1f}, "
        f"inside {inside.mean():.1%}, below D90 {below.mean():.1%}, above D10 {above.mean():.1%}"
    )
    # What a calibrated 80% band would need: quantiles of the DEMAND50 error.
    err = (g.actual - g.DEMAND50).groupby(by, observed=True)
    q = pd.DataFrame({"q10_err": err.quantile(0.1), "q90_err": err.quantile(0.9)})
    print("   empirical 10% / 90% quantiles of (actual - DEMAND50), MW, by lead:")
    print(q.round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--month", default="2025-01")
    args = ap.parse_args()
    cache = args.cache_dir / "nemweb"
    cache.mkdir(parents=True, exist_ok=True)
    month = pd.Period(args.month, freq="M")
    pd.set_option("display.width", 250)

    ident, df = read_region(download(cache, month))
    for c in ("RUN_DATETIME", "INTERVAL_DATETIME", "LASTCHANGED"):
        df[c] = nem_time(df[c])
    for c in ("DEMAND10", "DEMAND50", "DEMAND90", "DEMAND_AND_NONSCHEDGEN"):
        df[c] = pd.to_numeric(df[c])
    runs = structure(df, ident)

    main_type = df.RUNTYPE.value_counts().index[0]
    print(f"\n(analysis below uses RUNTYPE {main_type})")
    df = df.loc[df.RUNTYPE.eq(main_type)].copy()
    lc = runs.xs(main_type, level="RUNTYPE").lc
    df["lead_h"] = (df.INTERVAL_DATETIME - df.RUN_DATETIME.map(lc)).dt.total_seconds() / 3600
    df["lead_band"] = pd.cut(df.lead_h, [0, 0.5, 1, 2, 6, 12, 24, 48])

    start = df.RUN_DATETIME.min() - pd.Timedelta(hours=2)
    end = df.RUN_DATETIME.max() + pd.Timedelta(hours=2)
    pd_fc = predispatch(start, end)
    run_matching(df, pd_fc)
    dn = df.DEMAND_AND_NONSCHEDGEN - df.DEMAND50
    print("-- DEMAND_AND_NONSCHEDGEN - DEMAND50 (MW):", dn.describe().round(1).to_dict())
    tig = pd.to_numeric(df.TOTALINTERMITTENTGENERATION)
    print(f"   max |that - TOTALINTERMITTENTGENERATION|: {(dn - tig).abs().max():.3f} MW")
    band(df)
    df["actual"] = actual_mean30().reindex(df.INTERVAL_DATETIME).to_numpy()
    definition(df, pd_fc)
    against_actual(df)


if __name__ == "__main__":
    main()
