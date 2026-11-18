"""The frozen anomaly recipe (m4_plan.md step 4) and the checks that it holds.

The recipe is frozen **before the development run** (§1: no tuning step, so the development
verdicts are blind too). What is frozen (user 2026-11-19):

* the anomaly config, :func:`~gridcast.anomalies.recipe.anomaly_config_hash`, recorded as
  ``anomalies.frozen_config_hash``;
* the code, :data:`FROZEN_PATHS`, as at ``anomalies.frozen_commit``: M3's frozen paths
  (which hold M2's model and feature code), every ``anomalies`` module that scores, merges,
  injects or describes (:data:`RECIPE_MODULES`), and ``explainability/`` except its smoke
  module (:data:`EXPLAIN_MODULES`).

Not frozen: the smoke runs, the run and report modules that only call the frozen code and
store or read its outputs, and this module. A unit test classifies every module of both
packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gridcast.anomalies.recipe import anomaly_config_hash
from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.evaluation.holdout import git_state
from gridcast.intervals.freeze import FROZEN_PATHS as M3_FROZEN_PATHS

RECIPE_MODULES = (
    "detectors",
    "events",
    "evidence",
    "evidence_inputs",
    "injection",
    "inputs",
    "isolation",
    "profiles",
    "pvalues",
    "recipe",
    "reference",
    "run",
    "series",
)
EXPLAIN_MODULES = ("families", "pairs", "sample", "stored", "treeshap")
FROZEN_PATHS: tuple[str, ...] = (
    *M3_FROZEN_PATHS,
    *(f"src/gridcast/anomalies/{m}.py" for m in RECIPE_MODULES),
    *(f"src/gridcast/explainability/{m}.py" for m in EXPLAIN_MODULES),
)


@dataclass(frozen=True, slots=True)
class FreezeState:
    config_hash: str
    frozen_config_hash: str | None
    frozen_commit: str | None
    sha: str | None
    dirty: bool
    frozen_changed: tuple[str, ...]  # files under FROZEN_PATHS changed since the freeze

    @property
    def problems(self) -> list[str]:
        out = []
        if not (self.frozen_config_hash and self.frozen_commit):
            out.append("no freeze recorded (anomalies.frozen_*)")
        if self.frozen_config_hash and self.config_hash != self.frozen_config_hash:
            out.append(
                f"anomaly config hash {self.config_hash} differs from the frozen "
                f"{self.frozen_config_hash}"
            )
        if self.frozen_changed:
            out.append(f"frozen code changed since the freeze: {', '.join(self.frozen_changed)}")
        return out


def freeze_state(cfg: AppConfig, root: Path = PROJECT_ROOT) -> FreezeState:
    """The current anomaly hash and tree against the freeze record."""
    commit = cfg.anomalies.frozen_commit
    git = git_state(commit, root, FROZEN_PATHS)
    return FreezeState(
        config_hash=anomaly_config_hash(cfg),
        frozen_config_hash=cfg.anomalies.frozen_config_hash,
        frozen_commit=commit,
        sha=git.sha,
        dirty=git.dirty,
        frozen_changed=git.frozen_changed,
    )
