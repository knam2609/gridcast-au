"""The M2 holdout, scored once (criterion S5; m2_plan.md step 6).

``gridcast evaluate-holdout`` continues the frozen walk-forward recipe over the holdout:

* **Refusals first:** the ML config hash must equal ``backtest.frozen_ml_config_hash``; the
  working tree must be clean; the frozen code (:data:`FROZEN_PATHS`) must be unchanged
  since ``backtest.frozen_ml_commit``; run 2's development batches must be stored; and no
  holdout batch may exist for the config hash (the S5 lock). The data must cover every
  issue time in ``[holdout_start, holdout_end)`` with its full horizon.
* **The run** starts at the last development retrain (2025-08-01) and retrains quarterly
  through the holdout, exactly as the development walk-forward does. It fits the same three
  evaluation sets (``main``: A, C, D and the baselines; ``plus``: C⁺, D⁺; ``delay``: C with
  the longer GFS publication delay).
* **Reproduction check, before any scoring:** the first retrain's forecasts for development
  issue times must equal run 2's stored forecasts exactly (float32, NaN in the same places)
  for every ML model, and its fits must match run 2's stored fit records (setting, tree
  count, rows, every candidate's early-stopping MAE). Otherwise :class:`ReproductionFailed`
  is raised and nothing is scored or stored.
* **Scoring:** only then are the holdout issue times scored (baselines rescored on the
  same pairs) and stored as three batches, ``holdout_main``, ``holdout_plus`` and
  ``holdout_delay``. No holdout metric is computed before the check passes, and none is
  logged; ``gridcast report-holdout`` reads the stored results.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.evaluation.m2_results import artifact_path, load_evaluation_set, retrain_records
from gridcast.evaluation.report import TargetResults
from gridcast.evaluation.segments import weather_segments
from gridcast.evaluation.store import StoredModel, save_predictions, store_backtest
from gridcast.features.builder import FeatureBuilder
from gridcast.features.inputs import load_context, load_feature_builder
from gridcast.forecasting.backtest import evaluate
from gridcast.forecasting.ml_backtest import (
    MAIN_VARIANTS,
    PLUS_VARIANTS,
    TARGET,
    EvaluationSet,
    _baseline_predictions,
    _output,
    _records,
    follow_specs,
    grid_specs,
    ml_config_hash,
)
from gridcast.forecasting.origins import nem_midnight_utc, origin_positions
from gridcast.forecasting.walkforward import (
    ModelSpec,
    WalkForwardResult,
    model_name,
    retrain_schedule,
    run_walk_forward,
)
from gridcast.logs import get_logger
from gridcast.models.baselines import default_baselines
from gridcast.models.ml import FAMILIES
from gridcast.timeutils import NEM_TZ

log = get_logger(__name__)

HOLDOUT_SETS: dict[str, str] = {
    "main": "holdout_main",
    "plus": "holdout_plus",
    "delay": "holdout_delay",
}
# The model and feature code frozen with the config (m2_plan.md step 6).
FROZEN_PATHS = (
    "src/gridcast/features",
    "src/gridcast/models",
    "src/gridcast/weather",
    "src/gridcast/forecasting",
)
# Fields of a stored fit record that must reproduce; timings, file paths and the fold size
# (the holdout run's first fold also holds holdout issue times) are left out.
RECORD_FIELDS = (
    "retrain",
    "band",
    "cutoff",
    "rows",
    "es_train_rows",
    "es_valid_rows",
    "chosen",
    "params",
    "candidates",
)
VOLATILE_SCORE_FIELDS = frozenset({"seconds"})


class HoldoutRefused(RuntimeError):
    """A precondition of the single holdout run is not met; nothing was run."""


class ReproductionFailed(RuntimeError):
    """The holdout run does not reproduce run 2's development forecasts; nothing was scored."""


# --------------------------------------------------------------------------- preconditions


@dataclass(frozen=True, slots=True)
class GitState:
    sha: str | None
    dirty: bool
    frozen_changed: tuple[str, ...]  # files under FROZEN_PATHS differing from the frozen commit


def _git(args: Sequence[str], root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_state(
    frozen_commit: str | None, root: Path = PROJECT_ROOT, paths: Sequence[str] = FROZEN_PATHS
) -> GitState:
    """HEAD, whether tracked files are modified, and frozen files changed since the freeze.

    ``paths`` defaults to M2's frozen code; M3's interval freeze passes its own list.
    """
    try:
        sha = _git(["rev-parse", "HEAD"], root)
        dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], root))
        changed: set[str] = set()
        if frozen_commit:
            changed |= set(_git(["diff", "--name-only", frozen_commit, "--", *paths], root).split())
            untracked = _git(["ls-files", "--others", "--exclude-standard", "--", *paths], root)
            changed |= set(untracked.split())
    except (OSError, subprocess.CalledProcessError):
        return GitState(None, True, ())
    return GitState(sha, dirty, tuple(sorted(changed)))


@dataclass(frozen=True, slots=True)
class HoldoutWindow:
    first_retrain: pd.Timestamp  # the last development retrain
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp  # exclusive
    retrains: tuple[pd.Timestamp, ...]

    @property
    def first_label(self) -> str:
        return self.first_retrain.tz_convert(NEM_TZ).strftime("%Y-%m-%d")


def holdout_window(cfg: AppConfig) -> HoldoutWindow:
    if cfg.backtest.holdout_end is None:
        raise HoldoutRefused("backtest.holdout_end is not set")
    start = nem_midnight_utc(str(cfg.backtest.holdout_start))
    end = nem_midnight_utc(str(cfg.backtest.holdout_end))
    schedule = retrain_schedule(cfg.ml.first_issue, end, cfg.ml.retrain_every_months)
    development = [r for r in schedule if r < start]
    if not development:
        raise HoldoutRefused("no development retrain precedes the holdout")
    first = development[-1]
    plus_first = nem_midnight_utc(str(cfg.ml.plus_first_issue))
    if plus_first > first:
        raise HoldoutRefused("ml.plus_first_issue must not follow the last development retrain")
    return HoldoutWindow(first, start, end, tuple(r for r in schedule if r >= first))


def stored_holdout_batches(engine: Engine, region_id: str, config_hash: str) -> list[str]:
    """Holdout batches already stored for ``config_hash`` (the S5 lock)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT batch_id::text FROM model_runs
                WHERE region_id = :r AND target = 'demand'
                  AND config->>'evaluation_set' = ANY(:sets)
                  AND config->>'config_hash' = :h
                """
            ),
            {"r": region_id, "sets": list(HOLDOUT_SETS.values()), "h": config_hash},
        )
        return sorted(str(b) for b in rows.scalars())


def preflight(
    engine: Engine, cfg: AppConfig, region_id: str, git: GitState
) -> dict[str, TargetResults]:
    """Every refusal check; returns run 2's development batch per evaluation set."""
    frozen = cfg.backtest.frozen_ml_config_hash
    commit = cfg.backtest.frozen_ml_commit
    if frozen is None or commit is None:
        raise HoldoutRefused("no frozen ML config (backtest.frozen_ml_config_hash / _commit)")
    problems: list[str] = []
    current = ml_config_hash(cfg)
    if current != frozen:
        problems.append(f"ML config hash {current} differs from the frozen {frozen}")
    try:
        holdout_window(cfg)
    except HoldoutRefused as exc:
        problems.append(str(exc))
    if git.sha is None:
        problems.append("not a git checkout (the run's commit cannot be recorded)")
    elif git.dirty:
        problems.append("the working tree has uncommitted changes")
    if git.frozen_changed:
        problems.append(f"frozen code changed since {commit}: {', '.join(git.frozen_changed)}")
    if stored := stored_holdout_batches(engine, region_id, frozen):
        problems.append(
            f"the holdout is already scored for config {frozen} (batches {', '.join(stored)}); "
            "S5 allows one run"
        )
    references: dict[str, TargetResults] = {}
    for key in HOLDOUT_SETS:
        try:
            res = load_evaluation_set(engine, region_id, key)
        except LookupError as exc:
            problems.append(str(exc))
            continue
        hashes = {str(c.get("config_hash")) for c in res.runs["config"]}
        if hashes != {frozen}:
            problems.append(f"the latest '{key}' development batch is not the frozen config's")
        references[key] = res
    if problems:
        raise HoldoutRefused("; ".join(problems))
    return references


# --------------------------------------------------------------------------- reproduction


def same_instants(a: pd.DatetimeIndex, b: pd.DatetimeIndex) -> bool:
    """Element-wise equality, whatever the indexes' time units (pandas may read µs or ns)."""
    return len(a) == len(b) and bool((a == b).all())


@dataclass(slots=True)
class Reproduction:
    """Run 2's development forecasts and fit records, reproduced exactly (decision 3)."""

    evaluation_set: str
    reference_batch: str
    retrain: str
    models: list[str]
    issue_times: int  # development issue times compared
    pairs: int  # forecasts compared per model
    fits: int  # fit records compared (models x lead bands)
    mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def _normalise(record: Mapping[str, Any]) -> dict[str, Any]:
    """A fit record as stored (JSON round trip), without timings and paths."""
    stored = json.loads(json.dumps(dict(record), default=str))
    out = {k: stored[k] for k in RECORD_FIELDS}
    out["candidates"] = [
        {k: v for k, v in c.items() if k not in VOLATILE_SCORE_FIELDS} for c in out["candidates"]
    ]
    return out


def check_reproduction(
    key: str,
    result: WalkForwardResult,
    names: Sequence[str],
    reference: TargetResults,
    origin_times: pd.DatetimeIndex,
    development: np.ndarray,
    retrain: str,
    data_root: Path,
) -> Reproduction:
    """Compare the development part of ``result`` with run 2's stored batch, exactly."""
    times = origin_times[development]
    rep = Reproduction(
        key, str(reference.batch_id), retrain, list(names), len(times), 0, 0, mismatches=[]
    )
    for name in names:
        stored = pd.read_parquet(artifact_path(reference, name, data_root))
        stored_times = pd.DatetimeIndex(pd.to_datetime(stored["origin_time"], utc=True))
        sel = (stored_times >= times[0]) & (stored_times <= times[-1])
        if not same_instants(stored_times[sel], times):
            rep.mismatches.append(f"{name}: stored issue times differ from the run's")
            continue
        old = stored.loc[sel].drop(columns="origin_time").to_numpy(dtype=np.float32)
        new = result.predictions[name][development]
        rep.pairs = int(new.size)
        same = (old == new) | (np.isnan(old) & np.isnan(new))
        if not same.all():
            gap = np.nanmax(np.abs(old.astype(np.float64) - new.astype(np.float64)))
            rep.mismatches.append(
                f"{name}: {int((~same).sum()):,} of {same.size:,} forecasts differ "
                f"(max |diff| {gap:.6g} MW)"
            )
        old_fits = [
            _normalise(r) for r in retrain_records(reference, name) if r["retrain"] == retrain
        ]
        new_fits = [_normalise(asdict(r)) for r in result.retrains[name] if r.retrain == retrain]
        rep.fits += len(new_fits)
        if not old_fits:
            rep.mismatches.append(f"{name}: run 2 stored no fit for {retrain}")
        elif old_fits != new_fits:
            fields = sorted(
                {
                    k
                    for a, b in zip(old_fits, new_fits, strict=False)
                    for k in RECORD_FIELDS
                    if a[k] != b[k]
                }
            )
            rep.mismatches.append(f"{name}: {retrain} fit records differ ({', '.join(fields)})")
    log.info(
        "holdout.reproduction",
        evaluation_set=key,
        passed=rep.passed,
        models=len(names),
        issue_times=rep.issue_times,
        fits=rep.fits,
    )
    return rep


# --------------------------------------------------------------------------- the run


@dataclass(slots=True)
class HoldoutRun:
    config_hash: str
    window: HoldoutWindow
    sets: list[EvaluationSet]
    reproductions: list[Reproduction]


def _holdout_origins(builder: FeatureBuilder, window: HoldoutWindow) -> np.ndarray:
    """Issue times from the first retrain to the holdout end; every one must be scorable."""
    ctx, setup = builder.ctx, builder.setup
    origins = origin_positions(ctx, setup, window.first_retrain)
    origins = origins[ctx.times[origins] < window.holdout_end]
    expected = pd.date_range(
        window.first_retrain,
        window.holdout_end,
        freq=pd.Timedelta(minutes=5 * setup.issue_every_steps),
        inclusive="left",
    )
    got = pd.DatetimeIndex(ctx.times[origins])
    if not same_instants(got, expected):
        missing = expected.difference(got)
        first = missing[0] if len(missing) else "none"
        last_target = expected[-1] + setup.horizon_steps * pd.Timedelta(minutes=5)
        raise HoldoutRefused(
            f"{len(missing)} issue times lack data for their full horizon (first {first}); "
            f"the data end at {ctx.times[-1]}, the last holdout target is {last_target}"
        )
    return origins


def run_holdout(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    git: GitState,
    model_root: Path,
    references: Mapping[str, TargetResults] | None = None,
    plus: bool = True,
    delay: bool = True,
) -> HoldoutRun:
    """Fit, reproduce, then score the holdout (three evaluation sets; not stored here)."""
    refs = dict(references) if references is not None else preflight(engine, cfg, region_id, git)
    config_hash = ml_config_hash(cfg)
    window = holdout_window(cfg)
    ctx = load_context(engine, cfg, region_id)
    builder = load_feature_builder(engine, cfg, region_id, ctx=ctx)
    origins = _holdout_origins(builder, window)
    times = pd.DatetimeIndex(ctx.times[origins])
    development = np.asarray(times < window.holdout_start)
    holdout = ~development
    data_root = cfg.data_root()
    retrains = list(window.retrains)
    log.info(
        "holdout.start",
        first_retrain=window.first_label,
        retrains=len(retrains),
        development_issue_times=int(development.sum()),
        holdout_issue_times=int(holdout.sum()),
    )

    def batch_dir(batch: uuid.UUID) -> Path:
        return model_root / str(batch)

    def reproduce(key: str, res: WalkForwardResult, specs: Sequence[ModelSpec]) -> Reproduction:
        rep = check_reproduction(
            key,
            res,
            [s.name for s in specs],
            refs[key],
            times,
            development,
            window.first_label,
            data_root,
        )
        if not rep.passed:
            raise ReproductionFailed(
                f"[{key}] the holdout run does not reproduce run 2 ({rep.reference_batch}) at "
                f"{rep.retrain}; nothing scored or stored: " + "; ".join(rep.mismatches)
            )
        return rep

    # Fit every set and check it against run 2 before anything is scored.
    reps: list[Reproduction] = []

    main_batch = uuid.uuid4()
    main_specs = grid_specs(MAIN_VARIANTS, cfg)
    main = run_walk_forward(
        builder,
        main_specs,
        origins,
        retrains,
        cfg.ml,
        model_dir=batch_dir(main_batch),
        data_root=data_root,
    )
    reps.append(reproduce("main", main, main_specs))
    results: dict[str, tuple[uuid.UUID, WalkForwardResult, list[ModelSpec]]] = {
        "main": (main_batch, main, main_specs)
    }
    if plus:
        batch = uuid.uuid4()
        specs = grid_specs(PLUS_VARIANTS, cfg)
        res = run_walk_forward(
            builder,
            specs,
            origins,
            retrains,
            cfg.ml,
            train_start=nem_midnight_utc(str(cfg.ml.plus_train_start)),
            model_dir=batch_dir(batch),
            data_root=data_root,
        )
        reps.append(reproduce("plus", res, specs))
        results["plus"] = (batch, res, specs)
    delayed_builder: FeatureBuilder | None = None
    if delay:
        hours = cfg.features.delay_sensitivity_hours
        batch = uuid.uuid4()
        delayed_builder = load_feature_builder(engine, cfg, region_id, delay_hours=hours, ctx=ctx)
        specs = follow_specs("C", f"_delay{hours:g}h", main)
        res = run_walk_forward(
            delayed_builder,
            specs,
            origins,
            retrains,
            cfg.ml,
            model_dir=batch_dir(batch),
            data_root=data_root,
        )
        reps.append(reproduce("delay", res, specs))
        results["delay"] = (batch, res, specs)

    # Every reproduction passed: score the holdout issue times.
    assert builder.weather is not None and builder.weather.observed is not None
    observed = builder.weather.observed.series
    segments = weather_segments(
        ctx.local_day,
        ctx.frame["local_hour"].to_numpy(),
        observed["temp_weighted"],
        observed["cloud_weighted"],
        cfg.evaluation,
    )
    hold_origins = origins[holdout]
    baselines = _baseline_predictions(builder, hold_origins)
    folds = _folds(retrains, times[holdout], window.holdout_end)
    rep_by_key = {r.evaluation_set: r for r in reps}
    c_names = [model_name(f, "C") for f in FAMILIES]
    main_c = {n: main.predictions[n][holdout] for n in c_names}
    main_c_records = [r for r in _records(main, main_specs) if r.name in c_names]

    sets: list[EvaluationSet] = []
    for key, (batch, res, specs) in results.items():
        preds = {k: v[holdout] for k, v in res.predictions.items()}
        models: list[StoredModel] = list(_records(res, specs))
        extra: dict[str, Any] = {
            "train_rows": res.train_rows,
            "shared_train_rows": res.usable_rows,
        }
        if key == "main":
            preds |= baselines
            models += list(default_baselines(TARGET))
        elif key == "plus":  # C⁺/D⁺ next to C and the baselines, as in development
            preds |= main_c | baselines
            models += [*main_c_records, *default_baselines(TARGET)]
        else:  # delay: headline C next to C rebuilt with the later publication time
            preds = main_c | preds
            models = [*main_c_records, *models]
            extra["delay_hours"] = cfg.features.delay_sensitivity_hours
        out = _output(builder, hold_origins, preds, folds)
        metrics = evaluate(
            out, ctx, cfg.evaluation, window.holdout_start, segments, splits=("holdout",)
        )
        mask = out.common_mask()
        run_config: dict[str, Any] = {
            "evaluation_set": HOLDOUT_SETS[key],
            "period": "holdout",
            "config_hash": config_hash,
            "frozen_commit": cfg.backtest.frozen_ml_commit,
            "code_sha": git.sha,
            "setup": {
                "issue_every_steps": builder.setup.issue_every_steps,
                "horizon_steps": builder.setup.horizon_steps,
                "latency_steps": builder.setup.latency_steps,
            },
            "holdout_start": str(cfg.backtest.holdout_start),
            "holdout_end": str(cfg.backtest.holdout_end),
            "first_issue": str(out.origin_times[0]),
            "last_issue": str(out.origin_times[-1]),
            "complete_window": True,
            "ml": cfg.ml.model_dump(mode="json"),
            "lead_bands": [list(b) for b in cfg.ml.lead_bands],
            "publication_delay_hours": cfg.weather.forecasts.publication_delay_hours,
            "n_origins": len(out.origins),
            "retrains": out.folds,
            "scored_pairs": int(mask.sum()),
            "total_pairs": int(mask.size),
            "reproduction": rep_by_key[key].as_dict(),
            **extra,
        }
        sets.append(EvaluationSet(key, out, metrics, models, run_config, batch))
    return HoldoutRun(config_hash, window, sets, reps)


def _folds(
    retrains: Sequence[pd.Timestamp], times: pd.DatetimeIndex, end: pd.Timestamp
) -> list[str]:
    """NEM dates of the retrains whose fold holds at least one of ``times``."""
    labels = []
    for i, start in enumerate(retrains):
        stop = retrains[i + 1] if i + 1 < len(retrains) else end
        if ((times >= start) & (times < stop)).any():
            labels.append(start.tz_convert(NEM_TZ).strftime("%Y-%m-%d"))
    return labels


def store_holdout(
    engine: Engine, cfg: AppConfig, region_id: str, run: HoldoutRun, git: GitState
) -> dict[str, uuid.UUID]:
    """Store the holdout sets, re-checking the S5 lock first; records the run's start commit."""
    if stored := stored_holdout_batches(engine, region_id, run.config_hash):
        raise HoldoutRefused(f"holdout batches appeared during the run: {', '.join(stored)}")
    out: dict[str, uuid.UUID] = {}
    for es in run.sets:
        written = save_predictions(es.out, cfg.data_path(cfg.paths.backtest_dir), es.batch_id)
        artifacts = {name: path.relative_to(cfg.data_root()) for name, path in written.items()}
        store_backtest(
            engine,
            es.out,
            es.metrics,
            es.models,
            region_id,
            run.window.holdout_start,
            es.run_config,
            artifacts,
            es.batch_id,
            sha=git.sha,
        )
        out[es.key] = es.batch_id
    return out
