"""Persist backtest results: model_runs, forecast_metrics, latest forecasts, parquet."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.forecasting.backtest import BacktestOutput
from gridcast.provenance import git_sha


class StoredModel(Protocol):
    """What a ``model_runs`` row needs: baselines and walk-forward ML records both fit."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> str: ...

    def params(self) -> dict[str, object]: ...


def save_predictions(out: BacktestOutput, directory: Path, batch_id: uuid.UUID) -> dict[str, Path]:
    """Write one wide parquet per model (and the actuals): origin_time + h001…hHHH."""
    directory.mkdir(parents=True, exist_ok=True)
    cols = [f"h{h:03d}" for h in out.setup.horizons]
    paths: dict[str, Path] = {}
    matrices: dict[str, np.ndarray] = {"actual": out.actual.astype(np.float32), **out.predictions}
    for name, matrix in matrices.items():
        frame = pd.DataFrame(matrix, columns=cols)
        frame.insert(0, "origin_time", out.origin_times)
        path = directory / f"{out.target}__{name}__{batch_id}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        paths[name] = path
    return paths


def _metric_row(run_id: int, r: dict[Any, Any]) -> dict[str, Any]:
    value = float(r["value"])
    return {
        "run_id": run_id,
        "split": str(r["split"]),
        "segment": str(r["segment"]),
        "horizon": int(r["horizon"]),
        "metric": str(r["metric"]),
        "value": value if np.isfinite(value) else None,
        "n": int(r["n"]),
    }


def store_backtest(
    engine: Engine,
    out: BacktestOutput,
    metrics: pd.DataFrame,
    models: Sequence[StoredModel],
    region_id: str,
    holdout_start: pd.Timestamp,
    run_config: dict[str, Any],
    artifact_paths: dict[str, Path],
    batch_id: uuid.UUID,
    sha: str | None = None,
) -> dict[str, int]:
    """Insert one model_runs row per model plus its metrics and latest-origin forecasts.

    ``sha`` is the commit to record (a long run passes the one it started from); by default
    the current checkout's.
    """
    sha = sha or git_sha()
    run_ids: dict[str, int] = {}
    last = len(out.origins) - 1
    issue_time = out.origin_times[last].to_pydatetime()
    target_times = pd.DatetimeIndex(
        out.origin_times[last] + out.setup.horizons * pd.Timedelta(minutes=5)
    )
    with engine.begin() as conn:
        for model in models:
            config = {**run_config, "model_params": model.params()}
            run_id = conn.execute(
                text(
                    """
                    INSERT INTO model_runs
                        (batch_id, model_name, model_kind, target, region_id, eval_start,
                         eval_end, holdout_start, config, git_sha, artifact_path, status,
                         finished_at)
                    VALUES (:batch, :name, :kind, :target, :region, :eval_start, :eval_end,
                            :holdout, CAST(:config AS jsonb), :sha, :artifact, 'success', now())
                    RETURNING run_id
                    """
                ),
                {
                    "batch": batch_id,
                    "name": model.name,
                    "kind": model.kind,
                    "target": out.target,
                    "region": region_id,
                    "eval_start": out.origin_times[0].to_pydatetime(),
                    "eval_end": out.origin_times[-1].to_pydatetime(),
                    "holdout": holdout_start.to_pydatetime(),
                    "config": json.dumps(config, default=str),
                    "sha": sha,
                    "artifact": str(artifact_paths.get(model.name, "")) or None,
                },
            ).scalar_one()
            run_ids[model.name] = int(run_id)

            records = metrics.loc[metrics["model"] == model.name].to_dict(orient="records")
            conn.execute(
                text(
                    """
                    INSERT INTO forecast_metrics (run_id, split, segment, horizon, metric, value, n)
                    VALUES (:run_id, :split, :segment, :horizon, :metric, :value, :n)
                    """
                ),
                [_metric_row(run_id, r) for r in records],
            )

            point = out.predictions[model.name][last]
            conn.execute(
                text(
                    """
                    INSERT INTO forecasts
                        (run_id, region_id, issue_time, target_end, horizon, point)
                    VALUES (:run_id, :region, :issue, :target_end, :h, :point)
                    """
                ),
                [
                    {
                        "run_id": run_id,
                        "region": region_id,
                        "issue": issue_time,
                        "target_end": t.to_pydatetime(),
                        "h": int(h),
                        "point": float(p),
                    }
                    for h, t, p in zip(out.setup.horizons, target_times, point, strict=True)
                    if np.isfinite(p)
                ],
            )
    return run_ids
