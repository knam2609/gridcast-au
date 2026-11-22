"""``gridcast publish``: load the scored outputs into Postgres, exactly (m5_plan.md §2.3).

For each (product, split) source (:mod:`gridcast.serving.sources`):

1. Read the source matrix (intervals: CQR's bounds rearranged by the frozen function).
2. Compare any rows already stored for the published run with the source (the 288-row
   samples of M1/M2, or an earlier attempt). Every one must be equal, or nothing is written.
3. Month by month: binary COPY of the missing rows into a temporary table, then
   ``INSERT … ON CONFLICT DO NOTHING``, so a rerun inserts nothing and checks again.
4. **P1:** read every stored row of the run inside the split's window back (binary COPY, so
   floats come back as their bits) and compare it with the source: same pairs, the target
   at ``issue_time + horizon × 5 min``, and bit-identical values (points are stored as the
   float64 of their float32, bounds as float32). Missing pairs must be absent.
   **P4** (intervals): the share of stored pairs whose levels do not nest, in SQL.
5. **Interval metrics:** M3's :func:`gridcast.intervals.cqr_batch.cqr_metrics` on the
   bounds read back from Postgres and on the parquet bounds, both under M3's scoring mask
   (the point batch's common mask) with its actuals. They must be equal; they are then
   stored in ``forecast_metrics`` for the published interval run.
6. Record the publication (``publications``), with every check.

``--check-only`` does steps 1, 2, 4 and 5 on whatever is stored and writes nothing.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from sqlalchemy import Engine, text

from gridcast.config import SERVING_PRODUCTS, SERVING_SPLITS, AppConfig
from gridcast.forecasting.backtest import monthly_folds
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.batch import _metric_row
from gridcast.intervals.conformal import ConformalResult
from gridcast.intervals.cqr_batch import cqr_metrics
from gridcast.intervals.inputs import read_stored_batch
from gridcast.logs import get_logger
from gridcast.serving.pgcopy import RowLayout, copy_in, copy_out
from gridcast.serving.sources import (
    INTERVAL_LEVELS,
    INTERVAL_TABLE,
    Product,
    PublishRefused,
    Source,
    SourceMatrix,
    file_record,
    read_matrix,
    resolve_sources,
)

log = get_logger(__name__)

STAGE_TABLE = "serving_stage"
SQL_TYPE = {
    "timestamptz": "timestamptz",
    "smallint": "smallint",
    "real": "real",
    "double": "double precision",
}
PUBLISHED_EVALUATION_SET = "published"  # not one of M3's evaluation sets
TRANSFORM = (
    "gridcast.intervals.scores.rearrange_bounds: each pair's six CQR bounds sorted as "
    "quantiles, so the levels nest (M3's reported-only 'CQR, rearranged' row)"
)
MAX_EXAMPLES = 5


def value_kind(product: Product) -> str:
    """Wire type of the product's values: bounds are ``real``, points ``double precision``."""
    return "real" if product.kind == "interval" else "double"


def stage_layout(product: Product) -> RowLayout:
    kind = value_kind(product)
    return RowLayout(
        (
            ("issue_time", "timestamptz"),
            ("horizon", "smallint"),
            *((c, kind) for c in product.columns),
        )
    )


def read_layout(product: Product) -> RowLayout:
    kind = value_kind(product)
    return RowLayout(
        (
            ("issue_time", "timestamptz"),
            ("target_end", "timestamptz"),
            ("horizon", "smallint"),
            *((c, kind) for c in product.columns),
        )
    )


# --------------------------------------------------------------------------- rows


def long_rows(matrix: SourceMatrix, rows: np.ndarray) -> dict[str, np.ndarray]:
    """One row per (issue time, lead) with a value, for the issue rows ``rows``, in
    (issue time, lead) order: ``issue_time`` (Unix µs), ``horizon`` (1…H) and the values."""
    keep = ~matrix.missing[rows]
    i, h = np.nonzero(keep)
    out: dict[str, np.ndarray] = {
        "issue_time": matrix.times_us()[rows][i],
        "horizon": (h + 1).astype(np.int16),
    }
    for column, values in matrix.by_column.items():
        out[column] = values[rows][i, h]
    return out


@dataclass(frozen=True, slots=True)
class Chunk:
    label: str  # NEM month
    rows: np.ndarray  # issue rows of the matrix
    start: pd.Timestamp  # stored rows with issue times in [start, end) belong to it
    end: pd.Timestamp


def chunks(matrix: SourceMatrix, start: pd.Timestamp, end: pd.Timestamp) -> list[Chunk]:
    """NEM months of issue rows, with issue-time ranges that tile ``[start, end)`` so that a
    stored row at any time of the window is examined by exactly one chunk."""
    folds = monthly_folds(matrix.times)
    out = []
    for k, (label, rows) in enumerate(folds):
        lo = start if k == 0 else matrix.times[rows[0]]
        hi = end if k == len(folds) - 1 else matrix.times[folds[k + 1][1][0]]
        out.append(Chunk(label, rows, lo, hi))
    return out


def _driver(conn: Any) -> psycopg.Connection[Any]:
    return cast(psycopg.Connection[Any], conn.connection.driver_connection)


def insert_rows(
    engine: Engine,
    product: Product,
    run_id: int,
    region_id: str,
    rows: dict[str, np.ndarray],
    step_minutes: int,
) -> int:
    """Stage ``rows`` with binary COPY and insert those not stored yet; returns how many."""
    layout = stage_layout(product)
    if not len(rows["issue_time"]):
        return 0
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in product.columns)
    create = sql.SQL("CREATE TEMP TABLE {} ({}) ON COMMIT DROP").format(
        sql.Identifier(STAGE_TABLE),
        sql.SQL(", ").join(
            sql.SQL("{} {} NOT NULL").format(sql.Identifier(n), sql.SQL(SQL_TYPE[k]))
            for n, k in layout.columns
        ),
    )
    insert = sql.SQL(
        """
        INSERT INTO {table} (run_id, region_id, issue_time, target_end, horizon, {columns})
        SELECT %(run)s, %(region)s, issue_time,
               issue_time + make_interval(mins => horizon::integer * %(step)s::integer),
               horizon, {columns}
        FROM {stage}
        ON CONFLICT (run_id, issue_time, target_end) DO NOTHING
        """
    ).format(
        table=sql.Identifier(product.table), columns=columns, stage=sql.Identifier(STAGE_TABLE)
    )
    with engine.begin() as conn, _driver(conn).cursor() as cur:
        cur.execute(create)
        copy_in(cur, STAGE_TABLE, layout, rows)
        cur.execute(insert, {"run": run_id, "region": region_id, "step": step_minutes})
        return int(cur.rowcount)


# --------------------------------------------------------------------------- P1


@dataclass(slots=True)
class Comparison:
    """Stored rows of a run inside a window against the source matrix (P1)."""

    rows_expected: int = 0  # source pairs with a value
    rows_read: int = 0
    equal: int = 0  # stored rows equal to the source in every column, bit for bit
    value_mismatches: int = 0
    unexpected_rows: int = 0  # an issue time the source lacks, a lead outside 1…H, or a
    #                           pair the source leaves missing
    bad_target_end: int = 0
    missing_rows: int = 0  # source pairs with a value that are not stored
    examples: list[str] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        """Every stored row matches the source (some may still be missing)."""
        return self.value_mismatches == self.unexpected_rows == self.bad_target_end == 0

    @property
    def complete(self) -> bool:
        """Exact, and every source pair with a value is stored: P1 for this publication."""
        return self.exact and self.missing_rows == 0 and self.rows_read == self.rows_expected

    def add(self, other: Comparison) -> None:
        for name in (
            "rows_expected",
            "rows_read",
            "equal",
            "value_mismatches",
            "unexpected_rows",
            "bad_target_end",
            "missing_rows",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.examples += other.examples[: MAX_EXAMPLES - len(self.examples)]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "exact": self.exact, "complete": self.complete}

    def describe(self) -> str:
        parts = [f"{self.rows_read:,} of {self.rows_expected:,} rows stored"]
        for name in ("value_mismatches", "unexpected_rows", "bad_target_end"):
            if n := getattr(self, name):
                parts.append(f"{n:,} {name.replace('_', ' ')}")
        if self.examples:
            parts.append("e.g. " + "; ".join(self.examples))
        return ", ".join(parts)


def _bits(values: np.ndarray, kind: str) -> np.ndarray:
    """IEEE bits of the values as stored in a column of ``kind`` (float32 widened exactly)."""
    if kind == "double":
        return np.asarray(values, dtype=np.float64).view(np.uint64)
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def compare_rows(
    matrix: SourceMatrix,
    rows: np.ndarray,
    stored: dict[str, np.ndarray],
    kind: str,
    step_minutes: int,
    published: dict[str, np.ndarray] | None = None,
) -> Comparison:
    """Compare stored rows (decoded, any order) with the source's issue rows ``rows``.

    If ``published`` is given (full (n, H) float32 matrices), every stored row matching the
    source in pair and target is written into it, so it rebuilds what Postgres holds.
    """
    n_leads = matrix.n_leads
    t_us = matrix.times_us()[rows]
    expected = ~matrix.missing[rows]
    issue = stored["issue_time"]
    lead = stored["horizon"].astype(np.int64)
    n = len(issue)
    res = Comparison(rows_expected=int(expected.sum()), rows_read=n)

    pos = np.searchsorted(t_us, issue)
    pos_c = np.minimum(pos, max(len(t_us) - 1, 0))
    on_time = (pos < len(t_us)) & (t_us[pos_c] == issue) if len(t_us) else np.zeros(n, bool)
    in_lead = (lead >= 1) & (lead <= n_leads)
    col = np.where(in_lead, lead - 1, 0)
    known = on_time & in_lead
    pair = known & expected[pos_c, col] if len(t_us) else known
    res.unexpected_rows = int((~pair).sum())
    target_ok = stored["target_end"] == issue + lead * step_minutes * 60_000_000
    res.bad_target_end = int((~target_ok).sum())

    same = np.ones(n, dtype=bool)
    for column, values in matrix.by_column.items():
        src = values[rows][pos_c, col]
        same &= _bits(src, kind) == _bits(stored[column], kind)
    mismatch = pair & ~same
    res.value_mismatches = int(mismatch.sum())
    good = pair & same & target_ok
    res.equal = int(good.sum())

    flat = pos_c[pair] * n_leads + col[pair]
    if len(np.unique(flat)) != len(flat):  # impossible under the primary key
        raise ValueError("a pair is stored twice")
    present = np.zeros(expected.shape, dtype=bool)
    present.ravel()[flat] = True
    res.missing_rows = int((expected & ~present).sum())

    for k in np.flatnonzero(~(pair & same & target_ok))[:MAX_EXAMPLES]:
        when = pd.Timestamp(int(issue[k]), unit="us", tz="UTC").isoformat()
        what = "unexpected pair" if not pair[k] else "value" if not same[k] else "target_end"
        res.examples.append(f"{when} h{int(lead[k]):03d}: {what}")

    if published is not None:
        src_rows = rows[pos_c[good]]
        for column in matrix.by_column:
            published[column][src_rows, col[good]] = stored[column][good]
    return res


def read_stored(
    engine: Engine,
    product: Product,
    run_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, np.ndarray]:
    """Every stored row of ``run_id`` with an issue time in ``[start, end)`` (binary COPY)."""
    layout = read_layout(product)
    query = sql.SQL(
        "SELECT {cols} FROM {table} WHERE run_id = {run} AND issue_time >= {lo} "
        "AND issue_time < {hi} ORDER BY issue_time, target_end"
    ).format(
        cols=sql.SQL(", ").join(sql.Identifier(n) for n in layout.names),
        table=sql.Identifier(product.table),
        run=sql.Literal(run_id),
        lo=sql.Literal(start.to_pydatetime()),
        hi=sql.Literal(end.to_pydatetime()),
    )
    with engine.connect() as conn, _driver(conn).cursor() as cur:
        return copy_out(cur, query, layout)


def compare_stored(
    engine: Engine,
    source: Source,
    matrix: SourceMatrix,
    run_id: int | None,
    step_minutes: int,
    collect: bool = False,
) -> tuple[Comparison, dict[str, np.ndarray] | None]:
    """P1 over the whole window, month by month. ``collect`` also rebuilds the stored
    values as (n, H) float32 matrices (NaN where nothing is stored)."""
    published = (
        {c: np.full(v.shape, np.nan, dtype=np.float32) for c, v in matrix.by_column.items()}
        if collect
        else None
    )
    total = Comparison()
    kind = value_kind(source.product)
    for chunk in chunks(matrix, source.window.start, source.window.end):
        if run_id is None:
            stored = {n: np.empty(0, dtype=np.int64) for n in read_layout(source.product).names}
        else:
            stored = read_stored(engine, source.product, run_id, chunk.start, chunk.end)
        total.add(compare_rows(matrix, chunk.rows, stored, kind, step_minutes, published))
    return total, published


# --------------------------------------------------------------------------- SQL checks


def sql_checks(engine: Engine, source: Source, run_id: int, region_id: str) -> dict[str, Any]:
    """Region of every stored row, NULL interval columns of published points, and P4."""
    table = source.product.table
    window = {"r": run_id, "lo": source.window.start, "hi": source.window.end}
    out: dict[str, Any] = {}
    with engine.connect() as conn:
        out["other_region_rows"] = int(
            conn.execute(
                text(
                    f"SELECT count(*) FROM {table} WHERE run_id = :r AND issue_time >= :lo "
                    "AND issue_time < :hi AND region_id <> :region"
                ),
                {**window, "region": region_id},
            ).scalar_one()
        )
        if source.product.kind == "point":
            out["interval_columns_set"] = int(
                conn.execute(
                    text(
                        "SELECT count(*) FROM forecasts WHERE run_id = :r AND issue_time >= :lo "
                        "AND issue_time < :hi AND (lower IS NOT NULL OR upper IS NOT NULL "
                        "OR interval_level IS NOT NULL)"
                    ),
                    window,
                ).scalar_one()
            )
        else:
            row = conn.execute(
                text(
                    f"""
                    SELECT count(*) AS pairs,
                           count(*) FILTER (WHERE NOT (lo95 <= lo80 AND lo80 <= lo50
                               AND lo50 <= hi50 AND hi50 <= hi80 AND hi80 <= hi95)) AS non_nested
                    FROM {INTERVAL_TABLE} WHERE run_id = :r
                    """
                ),
                {"r": run_id},
            ).one()
            out["p4"] = {"pairs": int(row.pairs), "non_nested": int(row.non_nested)}
    return out


def sql_checks_pass(checks: dict[str, Any]) -> bool:
    ok = checks["other_region_rows"] == 0 and checks.get("interval_columns_set", 0) == 0
    if "p4" in checks:
        ok &= checks["p4"]["non_nested"] == 0 and checks["p4"]["pairs"] > 0
    return bool(ok)


# --------------------------------------------------------------------------- interval runs


def find_interval_run(engine: Engine, cfg: AppConfig, source: Source) -> int | None:
    """The published interval run built from ``source``'s run, if any."""
    with engine.connect() as conn:
        ids = (
            conn.execute(
                text(
                    """
                    SELECT run_id FROM model_runs
                    WHERE model_name = :name AND model_kind = 'interval' AND region_id = :r
                      AND config->>'evaluation_set' = :es
                      AND (config->>'source_run_id')::bigint = :src
                    ORDER BY run_id
                    """
                ),
                {
                    "name": cfg.serving.interval_model,
                    "r": source.run.region_id,
                    "es": PUBLISHED_EVALUATION_SET,
                    "src": source.run.run_id,
                },
            )
            .scalars()
            .all()
        )
    if len(ids) > 1:
        raise PublishRefused(f"{source.label}: several published interval runs {list(ids)}")
    return int(ids[0]) if ids else None


def interval_batch_id(cfg: AppConfig, source: Source) -> uuid.UUID:
    """Deterministic batch id of the published interval run of ``source``."""
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"gridcast-au/serving/{cfg.serving.interval_model}/{source.run.batch_id}",
    )


def create_interval_run(
    engine: Engine, cfg: AppConfig, source: Source, matrix: SourceMatrix, sha: str | None
) -> int:
    """The ``model_runs`` row of the rearranged bounds of one split (status ``running``)."""
    config = {
        "evaluation_set": PUBLISHED_EVALUATION_SET,
        "split": source.split,
        "source_run_id": source.run.run_id,
        "source_batch": source.run.batch_id,
        "source_model": source.run.model_name,
        "source_interval_config_hash": source.run.config.get("interval_config_hash"),
        "transform": TRANSFORM,
        "levels": list(INTERVAL_LEVELS),
        "scoring_batch": source.scoring_batch,
        "point_model": source.run.config.get("point_model"),
        "first_issue": str(matrix.times[0]),
        "last_issue": str(matrix.times[-1]),
    }
    with engine.begin() as conn:
        run_id = conn.execute(
            text(
                """
                INSERT INTO model_runs
                    (batch_id, model_name, model_kind, target, region_id, eval_start, eval_end,
                     holdout_start, config, git_sha, artifact_path, status)
                VALUES (:batch, :name, 'interval', 'demand', :region, :eval_start, :eval_end,
                        :holdout, CAST(:config AS jsonb), :sha, NULL, 'running')
                RETURNING run_id
                """
            ),
            {
                "batch": interval_batch_id(cfg, source),
                "name": cfg.serving.interval_model,
                "region": source.run.region_id,
                "eval_start": matrix.times[0].to_pydatetime(),
                "eval_end": matrix.times[-1].to_pydatetime(),
                "holdout": nem_midnight_utc(str(cfg.backtest.holdout_start)).to_pydatetime(),
                "config": json.dumps(config, default=str),
                "sha": sha,
            },
        ).scalar_one()
    return int(run_id)


def finish_interval_run(engine: Engine, run_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE model_runs SET status = 'success', finished_at = coalesce(finished_at, "
                "now()) WHERE run_id = :r"
            ),
            {"r": run_id},
        )


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True, slots=True)
class ScoringPairs:
    """M3's scoring inputs on the published issue times: actuals, mask and point forecast."""

    actual: np.ndarray
    mask: np.ndarray
    point: np.ndarray


def scoring_pairs(cfg: AppConfig, source: Source, times: pd.DatetimeIndex) -> ScoringPairs:
    """The stored point batch M3 scored this split on, restricted to ``times``: its actuals,
    its common mask (every model and the actual present) and the point model's forecasts."""
    if source.scoring_batch is None:
        raise ValueError(f"{source.label}: no scoring batch")
    batch = read_stored_batch(
        cfg.data_path(cfg.paths.backtest_dir), source.scoring_batch, source.product.target
    )
    rows = batch.origin_times.get_indexer(times)
    if np.any(rows < 0):
        raise ValueError(f"{source.label}: an issue time is not in the scoring batch")
    point_model = str(source.run.config.get("point_model"))
    if point_model not in batch.forecasts:
        raise ValueError(f"{source.label}: point model {point_model!r} not in the scoring batch")
    return ScoringPairs(
        batch.actual[rows], batch.common_mask()[rows], batch.forecasts[point_model][rows]
    )


def metrics_of(
    values: dict[str, np.ndarray], pairs: ScoringPairs, times: pd.DatetimeIndex
) -> pd.DataFrame:
    """M3's CQR metrics (``cqr_metrics``) of published bounds (float32 (n, H) by column)."""
    result = ConformalResult(  # only the bounds are read by cqr_metrics
        origin_times=times,
        lower={lv: values[f"lo{lv}"] for lv in INTERVAL_LEVELS},
        upper={lv: values[f"hi{lv}"] for lv in INTERVAL_LEVELS},
        candidates=[],
        selections=[],
        daily=pd.DataFrame(),
        clipped=pd.DataFrame(),
    )
    return cqr_metrics(result, pairs.actual, pairs.mask, pairs.point)


def stored_metrics(engine: Engine, run_id: int, split: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                "SELECT metric, horizon, value, n FROM forecast_metrics "
                "WHERE run_id = :r AND split = :s AND segment = 'all'"
            ),
            conn,
            params={"r": run_id, "s": split},
        )


def _metric_key(frame: pd.DataFrame) -> dict[tuple[str, int], tuple[float | None, int]]:
    values = frame["value"].to_numpy(dtype=np.float64, na_value=np.nan)
    keys = zip(frame["metric"].astype(str), frame["horizon"].astype(int), strict=True)
    return {
        key: (float(v) if np.isfinite(v) else None, int(n))
        for key, v, n in zip(keys, values, frame["n"].astype(int), strict=True)
    }


def check_metrics(
    engine: Engine,
    cfg: AppConfig,
    source: Source,
    matrix: SourceMatrix,
    published: dict[str, np.ndarray],
    run_id: int,
    write: bool,
) -> dict[str, Any]:
    """Interval metrics from Postgres vs from the parquet (must be equal), then vs what is
    stored for the published run (equal, or inserted if ``write`` and none are stored)."""
    pairs = scoring_pairs(cfg, source, matrix.times)
    from_db = metrics_of(published, pairs, matrix.times)
    from_parquet = metrics_of(matrix.by_column, pairs, matrix.times)
    equal = bool(from_db.equals(from_parquet))
    pooled_rows = from_db.loc[from_db["horizon"] == 0]
    pooled = dict(
        zip(pooled_rows["metric"], pooled_rows["value"].to_numpy(np.float64), strict=True)
    )
    out: dict[str, Any] = {
        "db_equals_parquet": equal,
        "metrics": len(from_db),
        "scored_pairs": int(pairs.mask.sum()),
        "pooled": {
            m: float(pooled[m])
            for m in (*(f"coverage_{lv}" for lv in INTERVAL_LEVELS), "non_nested")
        },
    }
    stored = stored_metrics(engine, run_id, source.split)
    if stored.empty:
        out["stored"] = "absent"
        if equal and write:
            records = from_db.assign(split=source.split, segment="all").to_dict(orient="records")
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO forecast_metrics
                            (run_id, split, segment, horizon, metric, value, n)
                        VALUES (:run_id, :split, :segment, :horizon, :metric, :value, :n)
                        """
                    ),
                    [_metric_row(run_id, r) for r in records],
                )
            out["stored"] = "inserted"
    else:
        out["stored"] = "equal" if _metric_key(stored) == _metric_key(from_db) else "differs"
    return out


def metrics_pass(checks: dict[str, Any] | None, write: bool) -> bool:
    if checks is None:
        return True
    want = {"inserted", "equal"} if write else {"equal"}
    return bool(checks["db_equals_parquet"]) and checks["stored"] in want


# --------------------------------------------------------------------------- publications


def start_publication(
    engine: Engine,
    source: Source,
    run_id: int,
    matrix: SourceMatrix,
    files: list[dict[str, Any]],
    sha: str | None,
) -> int:
    with engine.begin() as conn:
        pub = conn.execute(
            text(
                """
                INSERT INTO publications
                    (product, split, region_id, run_id, source_run_id, source_files,
                     first_issue, last_issue, issue_times, rows_expected, missing_pairs, git_sha)
                VALUES (:product, :split, :region, :run, :source_run, CAST(:files AS jsonb),
                        :first, :last, :n, :rows, :missing, :sha)
                RETURNING publication_id
                """
            ),
            {
                "product": source.product.name,
                "split": source.split,
                "region": source.run.region_id,
                "run": run_id,
                "source_run": source.run.run_id,
                "files": json.dumps(files),
                "first": matrix.times[0].to_pydatetime(),
                "last": matrix.times[-1].to_pydatetime(),
                "n": len(matrix.times),
                "rows": matrix.rows_expected,
                "missing": matrix.missing_pairs,
                "sha": sha,
            },
        ).scalar_one()
    return int(pub)


def finish_publication(
    engine: Engine,
    publication_id: int,
    status: str,
    rows_inserted: int,
    checks: dict[str, Any],
    error: str | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE publications
                SET status = :status, rows_inserted = :inserted, checks = CAST(:checks AS jsonb),
                    error = :error,
                    published_at = CASE WHEN :status = 'success' THEN now() END
                WHERE publication_id = :p
                """
            ),
            {
                "status": status,
                "inserted": rows_inserted,
                "checks": json.dumps(checks, default=str),
                "error": error,
                "p": publication_id,
            },
        )


# --------------------------------------------------------------------------- orchestration


@dataclass(slots=True)
class PublicationResult:
    label: str
    source_run_id: int
    run_id: int | None  # the published run (None: interval run not created yet)
    issue_times: int
    first_issue: pd.Timestamp
    last_issue: pd.Timestamp
    grid_gaps: int
    rows_expected: int
    missing_pairs: int
    before: Comparison  # stored rows found before writing
    after: Comparison | None = None  # P1 after writing (or on check-only)
    rows_inserted: int = 0
    sql: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] | None = None
    seconds: dict[str, float] = field(default_factory=dict)
    status: str = "running"
    publication_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def p1(self) -> bool:
        return self.after is not None and self.after.complete

    def checks(self) -> dict[str, Any]:
        return {
            "p1": self.after.as_dict() if self.after else None,
            "before": self.before.as_dict(),
            "grid_gaps": self.grid_gaps,
            "sql": self.sql,
            "interval_metrics": self.metrics,
            "seconds": self.seconds,
            **self.extra,
        }


class _Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.seconds[name] = round(now - self._t, 2)
        self._t = now


def publish_one(
    engine: Engine, cfg: AppConfig, source: Source, check_only: bool, sha: str | None
) -> PublicationResult:
    """Publish (or, with ``check_only``, check) one (product, split); see the module docstring."""
    n_leads = cfg.forecast.horizon_steps
    step = cfg.nem.dispatch_interval_minutes
    timer = _Timer()
    matrix = read_matrix(source, n_leads, cfg.forecast.issue_every_minutes)
    files = [file_record(f, source.data_root) for f in source.files]
    timer.lap("read_source")
    interval = source.product.kind == "interval"
    run_id = find_interval_run(engine, cfg, source) if interval else source.run.run_id
    before, published = compare_stored(engine, source, matrix, run_id, step, collect=interval)
    timer.lap("check_before")
    result = PublicationResult(
        source.label,
        source.run.run_id,
        run_id,
        len(matrix.times),
        matrix.times[0],
        matrix.times[-1],
        matrix.grid_gaps,
        matrix.rows_expected,
        matrix.missing_pairs,
        before,
        extra={"source": matrix.extra} if matrix.extra else {},
    )
    if not before.exact and not check_only:
        raise PublishRefused(
            f"{source.label}: stored rows differ from the source: {before.describe()}"
        )
    log.info(
        "publish.source",
        source=source.label,
        run=run_id,
        issue_times=len(matrix.times),
        rows=matrix.rows_expected,
        missing=matrix.missing_pairs,
        stored=before.rows_read,
    )

    if check_only:
        result.after = before
        if run_id is not None and before.rows_read:
            result.sql = sql_checks(engine, source, run_id, cfg.serving.region)
            if interval and before.complete and published is not None:
                result.metrics = check_metrics(
                    engine, cfg, source, matrix, published, run_id, write=False
                )
        timer.lap("checks")
        result.seconds = timer.seconds
        if not before.exact:
            result.status = "mismatch"
        elif (
            before.complete
            and result.sql
            and sql_checks_pass(result.sql)
            and metrics_pass(result.metrics, write=False)
        ):
            result.status = "complete"
        else:
            result.status = "incomplete"
        return result

    if run_id is None:
        run_id = create_interval_run(engine, cfg, source, matrix, sha)
        result.run_id = run_id
    pub_id = start_publication(engine, source, run_id, matrix, files, sha)
    result.publication_id = pub_id
    try:
        if not before.complete:
            for chunk in chunks(matrix, source.window.start, source.window.end):
                rows = long_rows(matrix, chunk.rows)
                n = insert_rows(engine, source.product, run_id, cfg.serving.region, rows, step)
                result.rows_inserted += n
                log.debug("publish.chunk", source=source.label, month=chunk.label, inserted=n)
            timer.lap("insert")
            after, published = compare_stored(
                engine, source, matrix, run_id, step, collect=interval
            )
            timer.lap("check_after")
        else:
            after = before  # nothing to insert: the check above is the read-back
        result.after = after
        result.sql = sql_checks(engine, source, run_id, cfg.serving.region)
        if interval and after.complete and published is not None:
            result.metrics = check_metrics(
                engine, cfg, source, matrix, published, run_id, write=True
            )
        timer.lap("checks")
        result.seconds = timer.seconds
        ok = after.complete and sql_checks_pass(result.sql) and metrics_pass(result.metrics, True)
        result.status = "success" if ok else "failed"
        finish_publication(
            engine,
            pub_id,
            result.status,
            result.rows_inserted,
            result.checks(),
            None if ok else "a check failed: " + after.describe(),
        )
        if ok and interval:
            finish_interval_run(engine, run_id)
    except Exception as exc:
        result.status = "failed"
        result.seconds = timer.seconds
        finish_publication(
            engine, pub_id, "failed", result.rows_inserted, result.checks(), repr(exc)
        )
        raise
    log.info("publish.done", source=source.label, status=result.status, **timer.seconds)
    return result


def publish(
    engine: Engine,
    cfg: AppConfig,
    check_only: bool,
    sha: str | None,
    products: Sequence[str] = SERVING_PRODUCTS,
    splits: Sequence[str] = SERVING_SPLITS,
) -> list[PublicationResult]:
    """Publish every requested (product, split), in order; stops at the first refusal.

    Writing needs a clean tree (``sha`` without ``-dirty``), so that the recorded commit
    describes the code that published.
    """
    if not check_only and (sha is None or sha.endswith("-dirty")):
        raise PublishRefused(f"publish needs a clean git tree (HEAD {sha})")
    sources = resolve_sources(engine, cfg, tuple(products), tuple(splits))
    results = [publish_one(engine, cfg, s, check_only, sha) for s in sources]
    if not check_only and any(r.rows_inserted for r in results):
        # Fresh planner statistics for the API's queries after a bulk load.
        with engine.begin() as conn:
            conn.execute(text(f"ANALYZE forecasts, {INTERVAL_TABLE}"))
    return results


def summary_rows(results: Sequence[PublicationResult]) -> list[list[Any]]:
    """One table row per publication for the CLI."""
    out = []
    for r in results:
        p4 = r.sql.get("p4")
        cov = (r.metrics or {}).get("pooled", {}).get("coverage_80")
        out.append(
            [
                r.label,
                r.source_run_id,
                r.run_id,
                f"{r.issue_times:,}",
                f"{r.rows_expected:,}",
                f"{r.missing_pairs:,}",
                f"{r.rows_inserted:,}",
                "pass" if r.p1 else "FAIL" if r.after and not r.after.exact else "incomplete",
                "" if p4 is None else ("pass" if p4["non_nested"] == 0 else f"FAIL {p4}"),
                "" if cov is None else f"{100 * cov:.2f}%",
                r.status,
                f"{sum(r.seconds.values()):.0f}",
            ]
        )
    return out


SUMMARY_HEADERS = [
    "product/split",
    "source run",
    "run",
    "issue times",
    "rows",
    "missing pairs",
    "inserted",
    "P1",
    "P4",
    "cov80",
    "status",
    "s",
]
