"""The frozen interval recipe (M3 step 5, m3_plan.md step 5) and the checks that it holds.

The recipe is frozen before the AEMO comparison and before the holdout, so neither can
influence it. What is frozen is the recipe, including the per-retrain selection rule, not
one set of fitted models:

* the interval config (``interval_config_hash``) and the QR config (``qr_config_hash``),
  recorded as ``intervals.frozen_config_hash`` and ``intervals.frozen_qr_config_hash``;
* the code that builds bounds, :data:`FROZEN_PATHS`, as at ``intervals.frozen_commit``:
  M2's frozen model and feature code, and every ``intervals`` module that produces bounds
  (5-minute and half-hour conformal, QR, CQR, and the rearrangement of the reported-only
  *CQR, rearranged* row). The report modules and this module are not frozen.

:func:`freeze_state` compares the current config and tree with the record.
:func:`reproduce_development` recomputes the stored step-1 conformal and step-3 CQR
development bounds with the current code and compares them exactly with the stored batches
(and their selections), which shows the frozen code is the code that built them.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import Engine

from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.evaluation.holdout import FROZEN_PATHS as M2_FROZEN_PATHS
from gridcast.evaluation.holdout import git_state
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.batch import PointTask, conformal_for, interval_config_hash
from gridcast.intervals.conformal import ConformalResult
from gridcast.intervals.cqr_batch import LevelTask, check_qr_batch, cqr_for, merge_levels
from gridcast.intervals.inputs import read_wide
from gridcast.intervals.qr_batch import qr_config_hash
from gridcast.intervals.report import BASELINES, CONFORMAL_C, HEADLINE, IntervalRun, find_runs
from gridcast.logs import get_logger

log = get_logger(__name__)

RECIPE_MODULES = (
    "batch",
    "conformal",
    "cqr_batch",
    "halfhour",
    "inputs",
    "qr_batch",
    "quantile",
    "rolling",
    "scores",
)
FROZEN_PATHS: tuple[str, ...] = (
    *M2_FROZEN_PATHS,
    *(f"src/gridcast/intervals/{m}.py" for m in RECIPE_MODULES),
)


@dataclass(frozen=True, slots=True)
class FreezeState:
    config_hash: str
    qr_config_hash: str
    frozen_config_hash: str | None
    frozen_qr_config_hash: str | None
    frozen_commit: str | None
    sha: str | None
    dirty: bool
    frozen_changed: tuple[str, ...]  # files under FROZEN_PATHS changed since the freeze

    @property
    def problems(self) -> list[str]:
        out = []
        if not (self.frozen_config_hash and self.frozen_qr_config_hash and self.frozen_commit):
            out.append("no freeze recorded (intervals.frozen_*)")
        if self.frozen_config_hash and self.config_hash != self.frozen_config_hash:
            out.append(
                f"interval config hash {self.config_hash} differs from the frozen "
                f"{self.frozen_config_hash}"
            )
        if self.frozen_qr_config_hash and self.qr_config_hash != self.frozen_qr_config_hash:
            out.append(
                f"QR config hash {self.qr_config_hash} differs from the frozen "
                f"{self.frozen_qr_config_hash}"
            )
        if self.frozen_changed:
            out.append(f"frozen code changed since the freeze: {', '.join(self.frozen_changed)}")
        return out


def freeze_state(cfg: AppConfig, root: Path = PROJECT_ROOT) -> FreezeState:
    """The current recipe hashes and tree against the freeze record."""
    commit = cfg.intervals.frozen_commit
    git = git_state(commit, root, FROZEN_PATHS)
    return FreezeState(
        config_hash=interval_config_hash(cfg),
        qr_config_hash=qr_config_hash(cfg),
        frozen_config_hash=cfg.intervals.frozen_config_hash,
        frozen_qr_config_hash=cfg.intervals.frozen_qr_config_hash,
        frozen_commit=commit,
        sha=git.sha,
        dirty=git.dirty,
        frozen_changed=git.frozen_changed,
    )


# --------------------------------------------------------------------------- reproduction


@dataclass(frozen=True, slots=True)
class Reproduction:
    run: str
    run_id: int
    batch_id: str
    git_sha: str | None
    issue_times: int
    bounds_equal: bool  # every level's stored bounds equal the recomputed ones exactly
    selections_equal: bool  # same choices at every (retrain, level, band)
    differing_bounds: tuple[str, ...]  # files that differ

    @property
    def passed(self) -> bool:
        return self.bounds_equal and self.selections_equal


def _compare(run: IntervalRun, result: ConformalResult, data_root: Path) -> Reproduction:
    directory = data_root / run.directory
    differing = []
    for level in result.lower:
        for side, values in (("lo", result.lower[level]), ("hi", result.upper[level])):
            name = f"{run.name}__{side}{level}.parquet"
            times, stored = read_wide(directory / name)
            if not times.equals(result.origin_times) or not np.array_equal(
                stored, values.astype(np.float64), equal_nan=True
            ):
                differing.append(name)
    fields = ("retrain", "level", "band", "chosen", "n_pairs")
    stored_sel = [tuple(s[f] for f in fields) for s in run.config.get("selections", [])]
    new_sel = [tuple(asdict(s)[f] for f in fields) for s in result.selections]
    return Reproduction(
        run=run.name,
        run_id=run.run_id,
        batch_id=run.batch_id,
        git_sha=run.git_sha,
        issue_times=len(result.origin_times),
        bounds_equal=not differing,
        selections_equal=stored_sel == new_sel,
        differing_bounds=tuple(differing),
    )


def reproduce_development(
    engine: Engine, cfg: AppConfig, region_id: str, workers: int = 4
) -> list[Reproduction]:
    """Recompute the stored development conformal and CQR bounds; compare them exactly."""
    runs = find_runs(engine, cfg, region_id)
    qr = check_qr_batch(engine, cfg)
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    backtest_dir = cfg.data_path(cfg.paths.backtest_dir)
    data_root = cfg.data_root()
    out = []

    names = {CONFORMAL_C: "lgbm_C", **{b: b.removeprefix("conformal_") for b in BASELINES}}
    tasks = [
        PointTask(cfg, region_id, backtest_dir, cfg.intervals.source_batch, model, end)
        for model in names.values()
    ]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        results = list(pool.map(conformal_for, tasks))
    for name, result in zip(names, results, strict=True):
        out.append(_compare(runs[name], result, data_root))
        log.info("intervals.freeze.reproduced", run=name, passed=out[-1].passed)

    levels = list(cfg.intervals.levels)
    level_tasks = [LevelTask(cfg, region_id, backtest_dir, qr.directory, lv, end) for lv in levels]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(level_tasks)))) as pool:
        parts = list(pool.map(cqr_for, level_tasks))
    out.append(_compare(runs[HEADLINE], merge_levels(parts, levels), data_root))
    log.info("intervals.freeze.reproduced", run=HEADLINE, passed=out[-1].passed)
    return out
