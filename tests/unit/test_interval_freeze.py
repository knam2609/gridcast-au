"""The interval recipe freeze (M3 step 5): what is frozen, and how differences are reported."""

from dataclasses import replace

from gridcast.config import PROJECT_ROOT, load_config
from gridcast.intervals.batch import interval_config_hash
from gridcast.intervals.freeze import FROZEN_PATHS, RECIPE_MODULES, FreezeState

# Modules that do not build bounds: reporting, and the freeze check itself.
UNFROZEN = {"__init__", "evaluation", "report", "freeze"}


def test_every_intervals_module_is_frozen_or_deliberately_not() -> None:
    """A new module must be classified here, so recipe code cannot slip past the freeze."""
    modules = {p.stem for p in (PROJECT_ROOT / "src/gridcast/intervals").glob("*.py")}
    assert modules == set(RECIPE_MODULES) | UNFROZEN
    assert not set(RECIPE_MODULES) & UNFROZEN
    assert all((PROJECT_ROOT / p).exists() for p in FROZEN_PATHS)
    assert "src/gridcast/models" in FROZEN_PATHS  # M2's frozen code stays frozen


def test_freeze_record_is_not_part_of_the_recipe_hash() -> None:
    cfg = load_config()
    base = interval_config_hash(cfg)
    cfg.intervals = cfg.intervals.model_copy(
        update={
            "frozen_config_hash": "x",
            "frozen_qr_config_hash": "y",
            "frozen_commit": "z",
        }
    )
    assert interval_config_hash(cfg) == base


def _state(**kw: object) -> FreezeState:
    base = FreezeState(
        config_hash="a",
        qr_config_hash="b",
        frozen_config_hash="a",
        frozen_qr_config_hash="b",
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
    assert "interval config hash a differs" in _state(frozen_config_hash="z").problems[0]
    assert "QR config hash b differs" in _state(frozen_qr_config_hash="z").problems[0]
    changed = _state(frozen_changed=("src/gridcast/intervals/conformal.py",)).problems
    assert changed == ["frozen code changed since the freeze: src/gridcast/intervals/conformal.py"]
