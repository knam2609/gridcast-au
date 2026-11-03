"""Generate ``reports/m2_feature_audit.md``: feature coverage measured on the real data.

Every number is computed here from the database, by the same builder that the models
will use, over every development issue time (every 30 minutes) and every horizon:

* pairs per variant, and how many have every feature (the pairs a model can score);
* which months lose pairs, and why (the GFS archive gap for C and C⁺);
* which forecast offset C uses, by lead time, with the configured publication delay and
  with the longer sensitivity delay;
* how far forecast temperature (C) is from ERA5 (D), by lead time;
* per-feature missing rates and ranges.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.features import catalogue
from gridcast.features.builder import FeatureBuilder, FeatureMatrix
from gridcast.features.inputs import load_context, load_weather_inputs
from gridcast.forecasting.backtest import monthly_folds
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc, origin_positions
from gridcast.logs import get_logger
from gridcast.provenance import git_sha
from gridcast.timeutils import NEM_TZ

log = get_logger(__name__)

REPORT_NAME = "m2_feature_audit.md"
# Lead-time bands in 5-minute steps (inclusive): 0–6 h, 6–12 h, 12–18 h, 18–24 h.
LEAD_BANDS: tuple[tuple[int, int, str], ...] = (
    (1, 72, "5 min – 6 h"),
    (73, 144, "6 h 5 min – 12 h"),
    (145, 216, "12 h 5 min – 18 h"),
    (217, 288, "18 h 5 min – 24 h"),
)


@dataclass(slots=True)
class ColumnStats:
    n: int = 0
    missing: int = 0
    total: float = 0.0
    lo: float = np.inf
    hi: float = -np.inf

    def update(self, values: np.ndarray) -> None:
        v = values.astype(np.float64)
        ok = np.isfinite(v)
        self.n += v.size
        self.missing += int((~ok).sum())
        if ok.any():
            self.total += float(v[ok].sum())
            self.lo = min(self.lo, float(v[ok].min()))
            self.hi = max(self.hi, float(v[ok].max()))

    @property
    def mean(self) -> float:
        present = self.n - self.missing
        return self.total / present if present else float("nan")


@dataclass(slots=True)
class MonthCoverage:
    pairs: int
    excluded: int
    first_target: pd.Timestamp | None  # first and last target of the excluded pairs
    last_target: pd.Timestamp | None


@dataclass(slots=True)
class VariantAudit:
    key: str  # variant id, plus a suffix for the sensitivity run
    label: str
    variant: str
    origins: int = 0
    pairs: int = 0
    complete: int = 0
    first_origin: pd.Timestamp | None = None
    last_origin: pd.Timestamp | None = None
    columns: dict[str, ColumnStats] = field(default_factory=dict)
    months: dict[str, MonthCoverage] = field(default_factory=dict)
    offsets: np.ndarray | None = None  # (n_bands, max_offset + 1) pair counts

    def update(self, fm: FeatureMatrix, month: str, origin_times: pd.DatetimeIndex) -> None:
        n_origins = len(np.unique(fm.origins))
        self.origins += n_origins
        self.pairs += len(fm.X)
        ok = fm.complete() & np.isfinite(fm.actual)
        self.complete += int(ok.sum())
        if self.first_origin is None:
            self.first_origin = origin_times[0]
        self.last_origin = origin_times[-1]
        for name in fm.X.columns:
            self.columns.setdefault(name, ColumnStats()).update(fm.X[name].to_numpy())
        bad = ~ok
        entry = MonthCoverage(len(fm.X), int(bad.sum()), None, None)
        if bad.any():
            times = origin_times.repeat(len(fm.X) // n_origins)[bad]
            lead = pd.to_timedelta(5 * fm.horizons[bad].astype(np.int64), unit="min")
            targets = times + lead
            entry.first_target, entry.last_target = targets.min(), targets.max()
        self.months[month] = entry
        if fm.weather_offset is not None:
            if self.offsets is None:
                self.offsets = np.zeros((len(LEAD_BANDS), 3), dtype=np.int64)
            for b, (lo, hi, _) in enumerate(LEAD_BANDS):
                in_band = (fm.horizons >= lo) & (fm.horizons <= hi)
                counts = np.bincount(fm.weather_offset[in_band], minlength=3)
                self.offsets[b] += counts[:3]


SHIFTS_H = (-1, 0, 1)  # timing check: ERA5 read this many hours after the target


@dataclass(slots=True)
class TemperatureGap:
    """Forecast (C) minus ERA5 (D) weighted temperature, by lead band, plus a timing check."""

    n: np.ndarray = field(default_factory=lambda: np.zeros(len(LEAD_BANDS), np.int64))
    abs_sum: np.ndarray = field(default_factory=lambda: np.zeros(len(LEAD_BANDS)))
    sum: np.ndarray = field(default_factory=lambda: np.zeros(len(LEAD_BANDS)))
    shift_n: np.ndarray = field(default_factory=lambda: np.zeros(len(SHIFTS_H), np.int64))
    shift_abs_sum: np.ndarray = field(default_factory=lambda: np.zeros(len(SHIFTS_H)))

    def update(self, c: FeatureMatrix, d: FeatureMatrix, era5: np.ndarray) -> None:
        fc = c.X["wx_temp"].to_numpy()
        diff = fc - d.X["wx_temp"].to_numpy()
        for b, (lo, hi, _) in enumerate(LEAD_BANDS):
            sel = (c.horizons >= lo) & (c.horizons <= hi) & np.isfinite(diff)
            self.n[b] += int(sel.sum())
            self.abs_sum[b] += float(np.abs(diff[sel]).sum())
            self.sum[b] += float(diff[sel].sum())
        # If the forecast and ERA5 were misaligned in time, the difference would be smallest
        # at a non-zero shift.
        for i, shift in enumerate(SHIFTS_H):
            pos = c.targets + 12 * shift
            ok = (pos >= 0) & (pos < len(era5)) & np.isfinite(fc)
            ok[ok] &= np.isfinite(era5[pos[ok]])
            self.shift_n[i] += int(ok.sum())
            self.shift_abs_sum[i] += float(np.abs(fc[ok] - era5[pos[ok]]).sum())

    @property
    def best_shift(self) -> int:
        return SHIFTS_H[int(np.argmin(self.shift_abs_sum / np.maximum(self.shift_n, 1)))]


@dataclass(slots=True)
class AuditResult:
    region: str
    generated: dt.datetime
    git: str | None
    data_first: pd.Timestamp
    data_last: pd.Timestamp
    dev_start: pd.Timestamp
    plus_start: pd.Timestamp
    holdout_start: pd.Timestamp
    delay_hours: float
    sensitivity_hours: float
    variants: dict[str, VariantAudit]
    temperature_gap: TemperatureGap


def _origins_by_month(
    times: pd.DatetimeIndex, positions: np.ndarray
) -> Iterable[tuple[str, np.ndarray, pd.DatetimeIndex]]:
    origin_times = pd.DatetimeIndex(times[positions])
    for label, idx in monthly_folds(origin_times):
        yield label, positions[idx], origin_times[idx]


def run_audit(engine: Engine, cfg: AppConfig, region_id: str) -> AuditResult:
    ctx = load_context(engine, cfg, region_id)
    setup = ForecastSetup.from_config(cfg)
    base = cfg.features.degree_base_c
    sens = cfg.features.delay_sensitivity_hours
    headline = FeatureBuilder(
        ctx, setup, load_weather_inputs(engine, cfg, region_id, ctx.times), base
    )
    delayed = FeatureBuilder(
        ctx, setup, load_weather_inputs(engine, cfg, region_id, ctx.times, sens), base
    )
    dev_start = nem_midnight_utc(str(cfg.ml.first_issue))
    plus_start = nem_midnight_utc(str(cfg.ml.plus_first_issue))
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    positions = origin_positions(ctx, setup, dev_start)
    positions = positions[ctx.times[positions] < holdout]
    if positions.size == 0:
        raise LookupError("no development issue times in the data (check ml.first_issue)")
    assert headline.weather is not None and headline.weather.observed is not None
    era5 = headline.weather.observed.series["temp_weighted"]

    audits = {v.id: VariantAudit(v.id, v.label, v.id) for v in catalogue.VARIANTS} | {
        "C_delay": VariantAudit("C_delay", f"C ({sens:g} h delay)", "C")
    }
    gap = TemperatureGap()
    for month, origins, origin_times in _origins_by_month(ctx.times, positions):
        built = {k: headline.build(k, origins) for k in ("A", "C", "D")}
        built["C_delay"] = delayed.build("C", origins)
        gap.update(built["C"], built["D"], era5)
        plus = origin_times >= plus_start
        if plus.any():
            for k in ("Cplus", "Dplus"):
                built[k] = headline.build(k, origins[plus])
        for key, fm in built.items():
            times = origin_times[plus] if key in ("Cplus", "Dplus") else origin_times
            audits[key].update(fm, month, times)
        log.info("feature_audit.month", month=month, origins=len(origins))

    return AuditResult(
        region=region_id,
        generated=dt.datetime.now(dt.UTC),
        git=git_sha(),
        data_first=ctx.times[0],
        data_last=ctx.times[-1],
        dev_start=dev_start,
        plus_start=plus_start,
        holdout_start=holdout,
        delay_hours=cfg.weather.forecasts.publication_delay_hours,
        sensitivity_hours=sens,
        variants=audits,
        temperature_gap=gap,
    )


# --------------------------------------------------------------------------- rendering


def _ts(value: pd.Timestamp | None) -> str:
    return "—" if value is None else value.strftime("%Y-%m-%d %H:%M UTC")


def _nem(value: pd.Timestamp) -> str:
    return value.tz_convert(NEM_TZ).strftime("%Y-%m-%d %H:%M")


def _int(v: int) -> str:
    return f"{v:,}"


def _pct(part: int, whole: int, digits: int = 2) -> str:
    return "—" if whole == 0 else f"{100 * part / whole:.{digits}f}%"


def _num(v: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(v) else f"{v:,.{digits}f}"


def _table(headers: list[str], rows: list[list[str]], align: str) -> str:
    sep = {"l": " :--- ", "r": " ---: "}
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep[a] for a in align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_audit(res: AuditResult) -> str:
    v = res.variants
    c, c_delay = v["C"], v["C_delay"]
    out = [
        "# M2 feature audit: coverage on the real data",
        "",
        f"> **Generated** by `gridcast feature-audit` on "
        f"{res.generated.strftime('%Y-%m-%d %H:%M UTC')} from the data in PostgreSQL, using "
        "the same feature builder as the models. Do not edit by hand. Feature definitions "
        "are in [`docs/feature_catalogue.md`](../docs/feature_catalogue.md).",
        "",
        "## Scope",
        "",
        _table(
            ["Item", "Value"],
            [
                ["Region", res.region],
                ["Demand data (interval end)", f"{_ts(res.data_first)} → {_ts(res.data_last)}"],
                [
                    "Development issue times (NEM time)",
                    f"{_nem(res.dev_start)} → before {_nem(res.holdout_start)}, every 30 min, "
                    "horizons 1–288",
                ],
                ["C⁺ / D⁺ issue times (NEM time)", f"from {_nem(res.plus_start)}"],
                [
                    "GFS publication delay",
                    f"{res.delay_hours:g} h (headline); {res.sensitivity_hours:g} h (sensitivity)",
                ],
                ["Git commit", f"`{res.git}`" if res.git else "—"],
            ],
            "ll",
        ),
        "",
        "## Pairs with every feature",
        "",
        "A pair (issue time, horizon) can be scored only if every feature of the variant "
        "and the actual demand are present. Pairs that some variant cannot score are "
        "excluded for every model and baseline when they are compared (m2_plan.md §2.4).",
        "",
        _table(
            ["Variant", "Issue times", "Pairs", "With every feature", "Excluded", "Share excluded"],
            [
                [
                    a.label,
                    _int(a.origins),
                    _int(a.pairs),
                    _int(a.complete),
                    _int(a.pairs - a.complete),
                    _pct(a.pairs - a.complete, a.pairs, 3),
                ]
                for a in v.values()
            ],
            "lrrrrr",
        ),
        "",
        "### Months with excluded pairs",
        "",
    ]
    rows = []
    for a in v.values():
        for month, m in a.months.items():
            if m.excluded:
                rows.append(
                    [
                        a.label,
                        month,
                        _int(m.excluded),
                        _pct(m.excluded, m.pairs),
                        _ts(m.first_target),
                        _ts(m.last_target),
                    ]
                )
    out += [
        _table(
            [
                "Variant",
                "Issue month (NEM)",
                "Excluded pairs",
                "Share of month",
                "First target",
                "Last target",
            ],
            rows,
            "llrrll",
        )
        if rows
        else "No variant excludes any pair.",
        "",
        "## Which forecast run C uses",
        "",
        "For each pair, C takes the freshest forecast offset whose values are published by "
        "the issue time (`docs/feature_catalogue.md`, *Forecast weather selection*). Shares "
        "of pairs by lead time:",
        "",
    ]
    offset_rows = []
    for a in (c, c_delay):
        assert a.offsets is not None
        bands = [(band, a.offsets[b]) for b, (_, _, band) in enumerate(LEAD_BANDS)]
        for band, counts in [*bands, ("**all leads**", a.offsets.sum(axis=0))]:
            day1, day2, none = int(counts[1]), int(counts[2]), int(counts[0])
            total = day1 + day2 + none
            offset_rows.append(
                [
                    a.label,
                    band,
                    _pct(day1, total, 1),
                    _pct(day2, total, 1),
                    _pct(none, total, 2),
                    _pct(day1, day1 + day2, 1),
                ]
            )
    out += [
        _table(
            [
                "Run",
                "Lead time",
                "Day 1",
                "Day 2",
                "None (excluded)",
                "Day 1 among pairs with weather",
            ],
            offset_rows,
            "llrrrr",
        ),
        "",
        "## Forecast temperature against ERA5",
        "",
        "Weighted temperature of C minus that of D, for the same pairs (both present). "
        "This is the weather-forecast error that separates C from its perfect-foresight "
        "mirror, plus the GFS-vs-ERA5 grid difference.",
        "",
        "The archive keeps only day-1 and day-2 forecasts (a day-0 run is never published "
        "before its valid time), so at **every** lead time C uses a forecast made 24–30 h "
        "(day 1) or 48–54 h (day 2) before the valid time. The difference therefore "
        "changes little with our lead time and grows only where day 2 takes over.",
        "",
    ]
    gap = res.temperature_gap
    gap_rows = [
        [
            band,
            _int(int(gap.n[b])),
            _num(gap.abs_sum[b] / gap.n[b], 2) if gap.n[b] else "—",
            _num(gap.sum[b] / gap.n[b], 2) if gap.n[b] else "—",
        ]
        for b, (_, _, band) in enumerate(LEAD_BANDS)
    ]
    n_all = int(gap.n.sum())
    gap_rows.append(
        [
            "**all leads**",
            _int(n_all),
            _num(gap.abs_sum.sum() / n_all, 2) if n_all else "—",
            _num(gap.sum.sum() / n_all, 2) if n_all else "—",
        ]
    )
    out += [
        _table(
            ["Lead time", "Pairs", "Mean absolute difference (°C)", "Mean difference (°C)"],
            gap_rows,
            "lrrr",
        ),
        "",
        "**Timing check.** The same difference with ERA5 read at the target time shifted "
        "by whole hours. A time-alignment error would make a non-zero shift fit better.",
        "",
        _table(
            ["ERA5 shift", "Pairs", "Mean absolute difference (°C)"],
            [
                [
                    f"{shift:+d} h",
                    _int(int(gap.shift_n[i])),
                    _num(gap.shift_abs_sum[i] / gap.shift_n[i], 3) if gap.shift_n[i] else "—",
                ]
                for i, shift in enumerate(SHIFTS_H)
            ],
            "lrr",
        ),
        "",
        f"Smallest difference at **{gap.best_shift:+d} h**"
        + (" (aligned)." if gap.best_shift == 0 else ": **investigate the alignment**."),
        "",
        "## Feature ranges and missing values",
        "",
        "Base and weather features from C over the development issue times, the extra "
        "weather features from C⁺ over its window, and the ERA5 weather features from D "
        "and D⁺.",
        "",
    ]
    stat_rows = []
    seen: set[str] = set()
    for key in ("C", "Cplus", "D", "Dplus"):
        a = v[key]
        for name, s in a.columns.items():
            label = name if key in ("C", "Cplus") else f"{name} (ERA5)"
            if label in seen or (key in ("D", "Dplus") and not name.startswith("wx_")):
                continue
            seen.add(label)
            stat_rows.append(
                [
                    f"`{label}`",
                    a.label,
                    _pct(s.missing, s.n, 3),
                    _num(s.lo, 2),
                    _num(s.mean, 2),
                    _num(s.hi, 2),
                ]
            )
    out += [
        _table(["Feature", "From", "Missing", "Min", "Mean", "Max"], stat_rows, "llrrrr"),
        "",
    ]
    return "\n".join(out)


def write_audit(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    result = run_audit(engine, cfg, region_id)
    path = cfg.resolve(cfg.paths.reports_dir) / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_audit(result), encoding="utf-8")
    return path
