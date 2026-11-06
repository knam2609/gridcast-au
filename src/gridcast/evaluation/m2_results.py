"""Load the stored M2 walk-forward results: one ``model_runs`` batch per evaluation set.

``gridcast backtest-ml`` stores three batches per run (``config.evaluation_set`` = ``main``,
``plus``, ``delay``). Smoke runs (``--until``) are never stored, but the loader also
requires ``complete_window`` so that only a full development run is ever reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, text

from gridcast.evaluation.report import TargetResults

EVALUATION_SETS = ("main", "plus", "delay")


def load_evaluation_set(engine: Engine, region_id: str, key: str) -> TargetResults:
    """The latest complete batch of evaluation set ``key`` (runs and metrics)."""
    if key not in EVALUATION_SETS:
        raise ValueError(f"unknown evaluation set {key!r}")
    with engine.connect() as conn:
        batch = conn.execute(
            text(
                """
                SELECT batch_id FROM model_runs
                WHERE region_id = :r AND target = 'demand' AND status = 'success'
                  AND config->>'evaluation_set' = :k
                  AND (config->>'complete_window')::boolean
                ORDER BY created_at DESC LIMIT 1
                """
            ),
            {"r": region_id, "k": key},
        ).scalar()
        if batch is None:
            raise LookupError(
                f"no complete '{key}' ML batch for {region_id}; run `gridcast backtest-ml`"
            )
        runs = pd.read_sql(
            text("SELECT * FROM model_runs WHERE batch_id = :b ORDER BY run_id"),
            conn,
            params={"b": batch},
        )
        metrics = pd.read_sql(
            text(
                """
                SELECT r.model_name AS model, m.split, m.segment, m.horizon, m.metric,
                       m.value, m.n
                FROM forecast_metrics m JOIN model_runs r USING (run_id)
                WHERE r.batch_id = :b
                """
            ),
            conn,
            params={"b": batch},
        )
    return TargetResults("demand", str(batch), runs, metrics)


def run_config(res: TargetResults, model: str) -> dict[str, Any]:
    """The stored ``config`` of one model of a batch."""
    sel = res.runs.loc[res.runs["model_name"] == model, "config"]
    if sel.empty:
        raise LookupError(f"batch {res.batch_id} has no model {model!r}")
    return dict(sel.iloc[0])


def retrain_records(res: TargetResults, model: str) -> list[dict[str, Any]]:
    """Per-(retrain, band) fit records of an ML model (``RetrainRecord`` as stored)."""
    return list(run_config(res, model)["model_params"]["retrains"])


def chosen_settings(res: TargetResults, model: str) -> dict[tuple[str, str], dict[str, Any]]:
    """``(retrain, band) -> chosen setting`` of an ML model."""
    return {(r["retrain"], r["band"]): dict(r["chosen"]) for r in retrain_records(res, model)}


def artifact_path(res: TargetResults, model: str, data_root: Path) -> Path:
    """Absolute path of a model's stored predictions (wide parquet)."""
    sel = res.runs.loc[res.runs["model_name"] == model, "artifact_path"]
    if sel.empty or not sel.iloc[0]:
        raise LookupError(f"batch {res.batch_id} has no predictions for {model!r}")
    return data_root / str(sel.iloc[0])


def actual_path(res: TargetResults, data_root: Path) -> Path:
    """The actuals parquet written next to the batch's predictions."""
    some = artifact_path(res, str(res.runs["model_name"].iloc[0]), data_root)
    return some.parent / f"demand__actual__{res.batch_id}.parquet"
