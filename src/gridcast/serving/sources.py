"""What ``gridcast publish`` loads, and the source matrices it checks against (m5_plan.md §2.1).

* :data:`PRODUCTS`: the four published products and the table each goes to.
* :func:`resolve_sources`: every (product, split) source run from ``serving.sources``,
  checked against its ``model_runs`` row before anything is written (:class:`PublishRefused`
  lists every problem at once).
* :func:`read_matrix`: a source's values on its split's issue times, as (issue time × lead)
  float32 matrices with NaN where nothing was stored (a *missing pair*, not published). The
  interval product is CQR's six stored bounds sorted per pair by the frozen
  :func:`gridcast.intervals.scores.rearrange_bounds`, as M3's reported-only row.

Splits by issue time: development from ``serving.first_issue`` to ``backtest.holdout_start``,
the holdout from there to ``backtest.holdout_end`` (NEM midnights).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import SERVING_PRODUCTS, SERVING_SPLITS, AppConfig
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.scores import rearrange_bounds

POINT_TABLE = "forecasts"
INTERVAL_TABLE = "forecast_intervals"
POINT_COLUMNS: tuple[str, ...] = ("point",)
# forecast_intervals' bound columns, in ascending quantile order (migration 0009).
INTERVAL_LEVELS: tuple[int, ...] = (50, 80, 95)
INTERVAL_COLUMNS: tuple[str, ...] = tuple(
    [f"lo{lv}" for lv in sorted(INTERVAL_LEVELS, reverse=True)]
    + [f"hi{lv}" for lv in sorted(INTERVAL_LEVELS)]
)
# Point products may come from these model kinds; the interval product from 'interval'.
POINT_KINDS = frozenset({"ml", "baseline"})
# Key of the point batch whose common mask M3 scored, in each split's interval run config.
SCORING_BATCH_KEY = {"dev": "source_batch", "holdout": "holdout_point_batch"}


@dataclass(frozen=True, slots=True)
class Product:
    name: str
    target: Literal["demand", "price"]
    kind: Literal["point", "interval"]

    @property
    def table(self) -> str:
        return INTERVAL_TABLE if self.kind == "interval" else POINT_TABLE

    @property
    def columns(self) -> tuple[str, ...]:
        return INTERVAL_COLUMNS if self.kind == "interval" else POINT_COLUMNS


PRODUCTS: dict[str, Product] = {
    "demand_point": Product("demand_point", "demand", "point"),
    "demand_interval": Product("demand_interval", "demand", "interval"),
    "price_persistence": Product("price_persistence", "price", "point"),
    "price_median_profile": Product("price_median_profile", "price", "point"),
}
if tuple(PRODUCTS) != SERVING_PRODUCTS:  # pragma: no cover - a coding error
    raise RuntimeError("serving products differ between the code and the config schema")


class PublishRefused(Exception):
    """A precondition failed; nothing was written."""


@dataclass(frozen=True, slots=True)
class Window:
    """Issue times ``[start, end)`` of one split (UTC)."""

    split: str
    start: pd.Timestamp
    end: pd.Timestamp


def windows(cfg: AppConfig) -> dict[str, Window]:
    first = nem_midnight_utc(str(cfg.serving.first_issue))
    hold = nem_midnight_utc(str(cfg.backtest.holdout_start))
    end = nem_midnight_utc(str(cfg.backtest.holdout_end))
    return {"dev": Window("dev", first, hold), "holdout": Window("holdout", hold, end)}


@dataclass(frozen=True, slots=True)
class SourceRun:
    """The ``model_runs`` row of a source run."""

    run_id: int
    batch_id: str
    model_name: str
    model_kind: str
    target: str
    region_id: str
    status: str
    artifact_path: str | None
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Source:
    """One (product, split) publication's source, resolved and checked."""

    product: Product
    window: Window
    run: SourceRun
    files: tuple[Path, ...]  # absolute; one per column for intervals (lo/hi per level)
    data_root: Path
    scoring_batch: str | None = None  # intervals: the point batch M3 scored on (common mask)

    @property
    def split(self) -> str:
        return self.window.split

    @property
    def label(self) -> str:
        return f"{self.product.name}/{self.split}"


def _fetch_run(engine: Engine, run_id: int) -> SourceRun | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT run_id, batch_id::text AS batch_id, model_name, model_kind, target,
                       region_id, status, artifact_path, config
                FROM model_runs WHERE run_id = :r
                """
            ),
            {"r": run_id},
        ).one_or_none()
    if row is None:
        return None
    return SourceRun(
        int(row.run_id),
        str(row.batch_id),
        row.model_name,
        row.model_kind,
        row.target,
        row.region_id,
        row.status,
        row.artifact_path,
        dict(row.config),
    )


def source_files(product: Product, run: SourceRun, data_root: Path) -> tuple[Path, ...]:
    """The parquet files of a source run: one wide file (points) or one per bound (intervals)."""
    if not run.artifact_path:
        return ()
    path = Path(run.artifact_path)
    path = path if path.is_absolute() else data_root / path
    if product.kind == "point":
        return (path,)
    return tuple(path / f"{run.model_name}__{column}.parquet" for column in INTERVAL_COLUMNS)


def resolve_sources(
    engine: Engine,
    cfg: AppConfig,
    products: tuple[str, ...] = SERVING_PRODUCTS,
    splits: tuple[str, ...] = SERVING_SPLITS,
) -> list[Source]:
    """Every requested (product, split) source, checked; refuses with every problem found.

    A source run must exist with status ``success`` and carry the configured batch id and
    model name, the product's target and kind and the served region, and its files must
    exist. An interval run must name, as the point batch it was scored on, the batch of the
    same split's published point forecasts.
    """
    if set(cfg.serving.interval_levels) != set(INTERVAL_LEVELS):
        raise PublishRefused(f"forecast_intervals holds levels {INTERVAL_LEVELS} only")
    data_root = cfg.data_root()
    wins = windows(cfg)
    problems: list[str] = []
    out: list[Source] = []
    for name in products:
        product = PRODUCTS[name]
        for split in splits:
            want = cfg.serving.sources[name][split]
            label = f"{name}/{split} (run {want.run_id})"
            run = _fetch_run(engine, want.run_id)
            if run is None:
                problems.append(f"{label}: no such run")
                continue
            checks = {
                "status": (run.status, "success"),
                "batch_id": (run.batch_id, want.batch_id),
                "model_name": (run.model_name, want.model_name),
                "target": (run.target, product.target),
                "region_id": (run.region_id, cfg.serving.region),
            }
            problems += [
                f"{label}: {key} is {got!r}, expected {exp!r}"
                for key, (got, exp) in checks.items()
                if got != exp
            ]
            kind_ok = (
                run.model_kind == "interval"
                if product.kind == "interval"
                else run.model_kind in POINT_KINDS
            )
            if not kind_ok:
                problems.append(f"{label}: model kind {run.model_kind!r} for a {product.kind}")
            files = source_files(product, run, data_root)
            if not files:
                problems.append(f"{label}: no artifact path")
            problems += [f"{label}: missing file {f}" for f in files if not f.is_file()]
            scoring = None
            if product.kind == "interval":
                scoring = run.config.get(SCORING_BATCH_KEY[split])
                point = cfg.serving.sources["demand_point"][split].batch_id
                if scoring != point:
                    problems.append(
                        f"{label}: scored on point batch {scoring!r}, but {split} publishes "
                        f"point batch {point!r}"
                    )
            out.append(Source(product, wins[split], run, files, data_root, scoring))
    if problems:
        raise PublishRefused("; ".join(problems))
    return out


# --------------------------------------------------------------------------- matrices


@dataclass(slots=True)
class SourceMatrix:
    """A source's values on its split's issue times: (n, H) float32 per column, NaN = missing."""

    times: pd.DatetimeIndex  # UTC issue times inside the window
    by_column: dict[str, np.ndarray]
    grid_gaps: int  # issue times absent between the first and last (on the issue grid)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_leads(self) -> int:
        return int(next(iter(self.by_column.values())).shape[1])

    @property
    def missing(self) -> np.ndarray:
        """Pairs without a value (every column is NaN there)."""
        return np.isnan(next(iter(self.by_column.values())))

    @property
    def missing_pairs(self) -> int:
        return int(self.missing.sum())

    @property
    def rows_expected(self) -> int:
        return int(self.missing.size) - self.missing_pairs

    def times_us(self) -> np.ndarray:
        """Issue times as int64 Unix microseconds."""
        return np.asarray(self.times.as_unit("us").asi8, dtype=np.int64)


def read_window(
    path: Path, window: Window, n_leads: int, issue_minutes: int
) -> tuple[pd.DatetimeIndex, np.ndarray, int]:
    """One wide parquet (``origin_time``, ``h001`` … ``hNNN``) on a window's issue times.

    Returns the UTC times, float64 values (as M3's ``read_wide``) and the number of grid gaps.
    Checks the lead columns, that times are UTC, strictly increasing and on the issue grid.
    """
    frame = pd.read_parquet(path)
    leads = [f"h{h:03d}" for h in range(1, n_leads + 1)]
    if list(frame.columns) != ["origin_time", *leads]:
        raise ValueError(f"{path.name}: expected origin_time and {leads[0]}…{leads[-1]}")
    times = pd.DatetimeIndex(frame.pop("origin_time"))
    if times.tz is None or str(times.tz) != "UTC":
        raise ValueError(f"{path.name}: origin_time must be UTC")
    keep = np.asarray((times >= window.start) & (times < window.end))
    times = times[keep]
    if len(times) == 0:
        raise ValueError(f"{path.name}: no issue time in {window.split}")
    step = pd.Timedelta(minutes=issue_minutes)
    if np.any(np.diff(times.as_unit("us").asi8) <= 0):
        raise ValueError(f"{path.name}: issue times are not strictly increasing")
    if np.any((times - times[0]) % step != pd.Timedelta(0)) or times[0].minute % issue_minutes:
        raise ValueError(f"{path.name}: an issue time is off the {issue_minutes}-minute grid")
    gaps = int((times[-1] - times[0]) / step) + 1 - len(times)
    values = frame.to_numpy(dtype=np.float64, na_value=np.nan)[keep]
    if np.any(np.isinf(values)):
        raise ValueError(f"{path.name}: infinite values")
    return times, values, gaps


def exact_float32(values: np.ndarray, name: str) -> np.ndarray:
    """``values`` as float32, refusing any value that float32 does not hold exactly."""
    out = values.astype(np.float32)
    ok = np.isfinite(values)
    if not np.array_equal(out[ok].astype(np.float64), values[ok]):
        raise ValueError(f"{name}: a value is not exactly representable as float32")
    return out


def read_matrix(source: Source, n_leads: int, issue_minutes: int) -> SourceMatrix:
    """The values ``gridcast publish`` writes for ``source`` (see the module docstring)."""
    if source.product.kind == "point":
        times, values, gaps = read_window(source.files[0], source.window, n_leads, issue_minutes)
        point = exact_float32(values, source.files[0].name)
        return SourceMatrix(times, {"point": point}, gaps)

    by_column: dict[str, np.ndarray] = {}
    times0: pd.DatetimeIndex | None = None
    gaps = 0
    for column, path in zip(INTERVAL_COLUMNS, source.files, strict=True):
        times, by_column[column], gaps = read_window(path, source.window, n_leads, issue_minutes)
        if times0 is None:
            times0 = times
        elif not times.equals(times0):
            raise ValueError(f"{path.name}: issue times differ from the other bounds")
    assert times0 is not None
    lower = {lv: by_column[f"lo{lv}"] for lv in INTERVAL_LEVELS}
    upper = {lv: by_column[f"hi{lv}"] for lv in INTERVAL_LEVELS}
    raw_missing = np.zeros(lower[INTERVAL_LEVELS[0]].shape, dtype=bool)
    for v in by_column.values():
        raw_missing |= np.isnan(v)
    # Pairs whose stored levels do not nest (M3's non_nested), which the sort repairs.
    non_nested = np.zeros(raw_missing.shape, dtype=bool)
    for lv_narrow, lv_wide in pairwise(INTERVAL_LEVELS):
        non_nested |= lower[lv_wide] > lower[lv_narrow]
        non_nested |= upper[lv_wide] < upper[lv_narrow]
    del by_column
    lo_r, hi_r = rearrange_bounds(lower, upper)
    del lower, upper
    values32 = {
        **{f"lo{lv}": exact_float32(lo_r[lv], f"lo{lv}") for lv in INTERVAL_LEVELS},
        **{f"hi{lv}": exact_float32(hi_r[lv], f"hi{lv}") for lv in INTERVAL_LEVELS},
    }
    ordered = {c: values32[c] for c in INTERVAL_COLUMNS}
    missing = np.isnan(ordered[INTERVAL_COLUMNS[0]])
    if any(not np.array_equal(np.isnan(v), missing) for v in ordered.values()):
        raise ValueError("rearranged bounds are missing at some levels only")
    if not np.array_equal(missing, raw_missing):
        raise ValueError("rearrangement changed which pairs are missing")
    return SourceMatrix(
        times0, ordered, gaps, {"non_nested_before": int(non_nested[~missing].sum())}
    )


def file_record(path: Path, data_root: Path) -> dict[str, Any]:
    """Path (relative to the data root where possible), SHA-256 and size of a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    shown = path.relative_to(data_root) if path.is_relative_to(data_root) else path
    return {"path": str(shown), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
