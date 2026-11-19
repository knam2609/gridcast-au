"""The anomaly recipe freeze (M4 step 4): what is frozen, and how differences are reported."""

from dataclasses import replace

import pytest

from gridcast.anomalies.freeze import (
    EXPLAIN_MODULES,
    FROZEN_PATHS,
    RECIPE_MODULES,
    FreezeState,
    freeze_state,
)
from gridcast.anomalies.recipe import anomaly_config_hash
from gridcast.config import PROJECT_ROOT, load_config
from gridcast.intervals.freeze import FROZEN_PATHS as M3_FROZEN_PATHS


def test_anomaly_recipe_is_frozen() -> None:
    """The recipe was frozen in step 4, before the development run (m4_plan.md §1): a setting
    change that alters any anomaly run fails here."""
    cfg = load_config()
    assert cfg.anomalies.frozen_commit == "2dc58fa"
    assert anomaly_config_hash(cfg) == cfg.anomalies.frozen_config_hash == "2ce8b3fc523ae81c"


def test_frozen_anomaly_code_is_unchanged() -> None:
    """No file under the frozen paths differs from the frozen commit."""
    state = freeze_state(load_config())
    if state.sha is None:
        pytest.skip("not a git checkout with the frozen commit")
    assert state.frozen_changed == ()
    assert state.problems == []


# Modules that only call the frozen code and store or read its outputs, the smoke runs, and
# the freeze check itself. Later stages (report, holdout) must be added here when written.
UNFROZEN_ANOMALIES = {"__init__", "dev", "diagnosis", "freeze", "report", "reproduce", "smoke"}
UNFROZEN_EXPLAIN = {"__init__", "smoke"}


def _modules(package: str) -> set[str]:
    return {p.stem for p in (PROJECT_ROOT / "src/gridcast" / package).glob("*.py")}


def test_every_anomalies_module_is_frozen_or_deliberately_not() -> None:
    """A new module must be classified here, so recipe code cannot slip past the freeze."""
    assert _modules("anomalies") == set(RECIPE_MODULES) | UNFROZEN_ANOMALIES
    assert not set(RECIPE_MODULES) & UNFROZEN_ANOMALIES
    assert _modules("explainability") == set(EXPLAIN_MODULES) | UNFROZEN_EXPLAIN
    assert not set(EXPLAIN_MODULES) & UNFROZEN_EXPLAIN


def test_frozen_paths_hold_m2_and_m3() -> None:
    assert set(M3_FROZEN_PATHS) <= set(FROZEN_PATHS)
    assert "src/gridcast/models" in FROZEN_PATHS
    assert "src/gridcast/intervals/rolling.py" in FROZEN_PATHS  # the p-value windows' base
    assert all((PROJECT_ROOT / p).exists() for p in FROZEN_PATHS)


def _state(**kw: object) -> FreezeState:
    base = FreezeState(
        config_hash="a",
        frozen_config_hash="a",
        frozen_commit="c0ffee",
        sha="d00d",
        dirty=False,
        frozen_changed=(),
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def test_freeze_state_problems() -> None:
    assert _state().problems == []
    assert _state(dirty=True).problems == []  # a dirty tree is the caller's concern
    assert "no freeze recorded" in _state(frozen_commit=None).problems[0]
    assert "anomaly config hash a differs" in _state(frozen_config_hash="z").problems[0]
    changed = _state(frozen_changed=("src/gridcast/anomalies/detectors.py",)).problems
    assert changed == ["frozen code changed since the freeze: src/gridcast/anomalies/detectors.py"]
