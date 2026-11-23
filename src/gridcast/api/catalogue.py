"""What the API serves, loaded from Postgres once (m5_plan.md §2.2, §2.4).

* **Publications:** for each (product, split), the latest successful ``publications`` row
  gives the run served and its issue-time range. ``/health`` separately checks that the
  *latest attempt* of each succeeded (step 2 question round, 2026-11-23).
* **Runs and metrics:** each published run's provenance (batch, git SHA, config hash) and its
  pooled metrics stored in ``forecast_metrics`` (segment ``all``, all leads).
* **Anomaly runs:** ``serving.anomaly_runs``, each checked to be a successful anomaly run of
  recipe v2 with the frozen v2 hash. Only their events are served.
* **Actuals window:** first and last stored interval end of the region.

Every connection is read-only (``default_transaction_read_only``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from gridcast.config import SERVING_PRODUCTS, SERVING_SPLITS, AppConfig
from gridcast.forecasting.origins import nem_midnight_utc

V2 = 2  # the anomaly recipe version served


def read_only_engine(url: str) -> Engine:
    """A pooled engine whose sessions run in UTC and cannot write."""
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=UTC -c default_transaction_read_only=on"},
    )


def _ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC").as_unit("us")


@dataclass(frozen=True, slots=True)
class Published:
    product: str
    split: str
    run_id: int
    source_run_id: int
    first_issue: pd.Timestamp
    last_issue: pd.Timestamp
    issue_times: int
    rows: int
    missing_pairs: int
    publication_id: int
    published_at: pd.Timestamp | None

    def covers(self, t: pd.Timestamp) -> bool:
        return bool(self.first_issue <= t <= self.last_issue)


@dataclass(frozen=True, slots=True)
class RunInfo:
    run_id: int
    batch_id: str
    model_name: str
    model_kind: str
    status: str
    git_sha: str | None
    config: dict[str, Any]

    @property
    def config_hash(self) -> str | None:
        for key in (
            "config_hash",
            "interval_config_hash",
            "source_interval_config_hash",
            "anomaly_config_hash",
        ):
            if value := self.config.get(key):
                return str(value)
        return None


@dataclass(frozen=True, slots=True)
class Metrics:
    values: dict[str, float | None]
    n: int | None


@dataclass(frozen=True, slots=True)
class AnomalyRunInfo:
    split: str
    run: RunInfo
    events: int
    first_event: pd.Timestamp | None
    last_event: pd.Timestamp | None


@dataclass(slots=True)
class Catalogue:
    region: str
    first_issue: pd.Timestamp  # serving.first_issue (NEM midnight)
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    published: dict[tuple[str, str], Published]
    runs: dict[int, RunInfo]
    metrics: dict[tuple[int, str], Metrics]
    anomaly_runs: dict[str, AnomalyRunInfo]
    anomaly_problems: list[str] = field(default_factory=list)
    actuals: tuple[pd.Timestamp, pd.Timestamp] | None = None

    def split_of(self, t: pd.Timestamp) -> str:
        return "dev" if t < self.holdout_start else "holdout"

    def get(self, product: str, split: str) -> Published | None:
        return self.published.get((product, split))

    def issue_range(self, product: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        pubs = [p for (prod, _), p in self.published.items() if prod == product]
        if not pubs:
            return None
        return min(p.first_issue for p in pubs), max(p.last_issue for p in pubs)

    @property
    def anomaly_run_ids(self) -> list[int]:
        return [a.run.run_id for a in self.anomaly_runs.values()]


PUBLICATION_COLUMNS = """
    publication_id, product, split, run_id, source_run_id, first_issue, last_issue,
    issue_times, rows_expected, missing_pairs, status, published_at
"""


def latest_attempts(engine: Engine, region: str) -> dict[tuple[str, str], str]:
    """Status of the latest publication attempt of each (product, split)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT ON (product, split) product, split, status
                FROM publications WHERE region_id = :r
                ORDER BY product, split, started_at DESC, publication_id DESC
                """
            ),
            {"r": region},
        ).all()
    return {(r.product, r.split): r.status for r in rows}


def _runs(engine: Engine, ids: set[int]) -> dict[int, RunInfo]:
    if not ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT run_id, batch_id::text AS batch_id, model_name, model_kind, status,
                       git_sha, config
                FROM model_runs WHERE run_id = ANY(:ids)
                """
            ),
            {"ids": sorted(ids)},
        ).all()
    return {
        int(r.run_id): RunInfo(
            int(r.run_id),
            r.batch_id,
            r.model_name,
            r.model_kind,
            r.status,
            r.git_sha,
            dict(r.config),
        )
        for r in rows
    }


def _metrics(engine: Engine, keys: set[tuple[int, str]]) -> dict[tuple[int, str], Metrics]:
    if not keys:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT run_id, split, metric, value, n FROM forecast_metrics
                WHERE run_id = ANY(:ids) AND segment = 'all' AND horizon = 0
                ORDER BY run_id, split, metric
                """
            ),
            {"ids": sorted({k[0] for k in keys})},
        ).all()
    values: dict[tuple[int, str], dict[str, float | None]] = {}
    counts: dict[tuple[int, str], set[int]] = {}
    for r in rows:
        key = (int(r.run_id), r.split)
        if key not in keys:
            continue
        values.setdefault(key, {})[r.metric] = None if r.value is None else float(r.value)
        counts.setdefault(key, set()).add(int(r.n))
    return {
        k: Metrics(v, next(iter(counts[k])) if len(counts[k]) == 1 else None)
        for k, v in values.items()
    }


def _anomaly_runs(engine: Engine, cfg: AppConfig) -> tuple[dict[str, AnomalyRunInfo], list[str]]:
    want = cfg.serving.anomaly_runs
    runs = _runs(engine, {w.run_id for w in want.values()})
    frozen = cfg.anomalies_v2.frozen_config_hash
    problems: list[str] = []
    out: dict[str, AnomalyRunInfo] = {}
    for split in SERVING_SPLITS:
        w = want[split]
        run = runs.get(w.run_id)
        label = f"anomaly run {w.run_id} ({split})"
        if run is None:
            problems.append(f"{label}: not found")
            continue
        found = {
            "status": (run.status, "success"),
            "batch_id": (run.batch_id, w.batch_id),
            "model_name": (run.model_name, w.model_name),
            "recipe_version": (run.config.get("recipe_version"), V2),
            "anomaly_config_hash": (run.config.get("anomaly_config_hash"), frozen),
            "split": (run.config.get("split"), split),
        }
        bad = [f"{k} is {g!r}, expected {e!r}" for k, (g, e) in found.items() if g != e]
        if bad:
            problems.append(f"{label}: " + ", ".join(bad))
            continue
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT count(*) AS n, min(first_interval_end) AS lo, "
                    "max(first_interval_end) AS hi FROM anomalies WHERE run_id = :r"
                ),
                {"r": run.run_id},
            ).one()
        out[split] = AnomalyRunInfo(
            split,
            run,
            int(row.n),
            None if row.lo is None else _ts(row.lo),
            None if row.hi is None else _ts(row.hi),
        )
    return out, problems


def load_catalogue(engine: Engine, cfg: AppConfig) -> Catalogue:
    region = cfg.serving.region
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT DISTINCT ON (product, split) {PUBLICATION_COLUMNS}
                FROM publications WHERE region_id = :r AND status = 'success'
                ORDER BY product, split, started_at DESC, publication_id DESC
                """
            ),
            {"r": region},
        ).all()
        window = conn.execute(
            text(
                "SELECT min(interval_end) AS lo, max(interval_end) AS hi FROM market_intervals "
                "WHERE region_id = :r"
            ),
            {"r": region},
        ).one()
    published = {
        (r.product, r.split): Published(
            r.product,
            r.split,
            int(r.run_id),
            int(r.source_run_id),
            _ts(r.first_issue),
            _ts(r.last_issue),
            int(r.issue_times),
            int(r.rows_expected),
            int(r.missing_pairs),
            int(r.publication_id),
            None if r.published_at is None else _ts(r.published_at),
        )
        for r in rows
        if r.product in SERVING_PRODUCTS
    }
    ids = {p.run_id for p in published.values()} | {p.source_run_id for p in published.values()}
    anomaly_runs, problems = _anomaly_runs(engine, cfg)
    return Catalogue(
        region=region,
        first_issue=nem_midnight_utc(str(cfg.serving.first_issue)),
        holdout_start=nem_midnight_utc(str(cfg.backtest.holdout_start)),
        holdout_end=nem_midnight_utc(str(cfg.backtest.holdout_end)),
        published=published,
        runs=_runs(engine, ids),
        metrics=_metrics(engine, {(p.run_id, p.split) for p in published.values()}),
        anomaly_runs=anomaly_runs,
        anomaly_problems=problems,
        actuals=None if window.lo is None else (_ts(window.lo), _ts(window.hi)),
    )
