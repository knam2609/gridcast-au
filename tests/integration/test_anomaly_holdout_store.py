"""The holdout run's refusals and A6 lock (m4_plan.md §1, "Clarified in step 6")."""

import datetime as dt
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from gridcast.anomalies.dev import store_dev
from gridcast.anomalies.freeze import FreezeState, freeze_state
from gridcast.anomalies.holdout import (
    HoldoutRefused,
    HoldoutRun,
    Preflight,
    preflight,
    store_holdout,
)
from gridcast.anomalies.reproduce import Reproduction
from gridcast.config import load_config
from gridcast.database.load import ensure_regions
from tests.integration.test_anomaly_dev_store import _outputs

pytestmark = pytest.mark.integration


def _state(version: int) -> FreezeState:
    """The real hash, as preflight computes it, with a clean frozen record."""
    real = freeze_state(load_config(), version=version)
    return FreezeState(real.config_hash, real.config_hash, "abc1234", "abc1234ff", False, ())


def _store(db: Engine, tmp_path: Path, split: str, version: int = 1) -> int:
    cfg = load_config()
    cfg.anomalies = cfg.anomalies.model_copy(update={"output_dir": tmp_path})
    out = _outputs()
    out = replace(out, split=replace(out.split, name=split))
    run_id, _, _ = store_dev(db, cfg, "VIC1", out, _state(version))
    return run_id


def test_no_development_run_refuses(db: Engine) -> None:
    with db.begin() as conn:
        ensure_regions(conn, load_config())
    with pytest.raises(HoldoutRefused, match="expected one development run"):
        preflight(db, load_config(), "VIC1", 1)


def test_a_stored_holdout_run_refuses_a_second_a6(db: Engine, tmp_path: Path) -> None:
    with db.begin() as conn:
        ensure_regions(conn, load_config())
    _store(db, tmp_path / "dev", "dev")
    try:
        preflight(db, load_config(), "VIC1", 1)
    except HoldoutRefused as exc:  # a dirty tree while developing is fine here
        assert "scored once (A6)" not in str(exc)
    held = _store(db, tmp_path / "holdout", "holdout")
    with pytest.raises(HoldoutRefused, match=rf"run {held}, .*scored once \(A6\)"):
        preflight(db, load_config(), "VIC1", 1)
    # The lock is per recipe hash: v2 is not locked by v1's holdout run.
    with pytest.raises(HoldoutRefused) as v2:
        preflight(db, load_config(), "VIC1", 2)
    assert "A6" not in str(v2.value) and "expected one development run" in str(v2.value)


def test_store_checks_the_lock_again(db: Engine, tmp_path: Path) -> None:
    with db.begin() as conn:
        ensure_regions(conn, load_config())
    _store(db, tmp_path / "dev", "dev")
    held = _store(db, tmp_path / "holdout", "holdout")
    cfg = load_config()
    cfg.anomalies = cfg.anomalies.model_copy(update={"output_dir": tmp_path / "second"})
    out = _outputs()
    out = replace(out, split=replace(out.split, name="holdout"))
    rep = Reproduction(1, 1, "b", None, False, [], 0.0)
    run = HoldoutRun(Preflight(1, _state(1), 1, "b"), rep, [], out)
    with pytest.raises(HoldoutRefused, match=f"run {held}"):
        store_holdout(db, cfg, "VIC1", run)
    with db.connect() as conn:
        n = conn.execute(
            text("SELECT count(*) FROM model_runs WHERE config->>'split' = 'holdout'")
        ).scalar_one()
    assert n == 1
    assert not (tmp_path / "second").exists()


def test_the_holdout_record_names_both_batches_and_the_reproduction(
    db: Engine, tmp_path: Path
) -> None:
    with db.begin() as conn:
        ensure_regions(conn, load_config())
    cfg = load_config()
    cfg.anomalies = cfg.anomalies.model_copy(update={"output_dir": tmp_path})
    out = _outputs()
    out = replace(out, split=replace(out.split, name="holdout"))
    rep = Reproduction(1, 104, "b", None, False, [], 3.0)
    run = HoldoutRun(Preflight(1, _state(1), 104, "b"), rep, [], out)
    run_id, _, _ = store_holdout(db, cfg, "VIC1", run)
    with db.connect() as conn:
        config = conn.execute(
            text("SELECT config FROM model_runs WHERE run_id = :r"), {"r": run_id}
        ).scalar_one()
        splits = (
            conn.execute(
                text("SELECT DISTINCT split FROM anomalies WHERE run_id = :r"), {"r": run_id}
            )
            .scalars()
            .all()
        )
    a = cfg.anomalies
    assert config["split"] == "holdout" and splits == ["holdout"]
    assert config["inputs"]["qr_batches"] == [
        a.residual.dev_qr_batch,
        a.residual.holdout_qr_batch,
    ]
    assert config["inputs"]["mask_batches"] == [a.explain.dev_batch, a.explain.holdout_batch]
    assert config["reproduction"]["development_run"] == 104
    assert (
        dt.date.fromisoformat(config["periods"]["market_first_day"]) == out.split.market_first_day
    )
