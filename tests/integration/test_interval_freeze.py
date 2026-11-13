"""M3 step 5: the freeze check recomputes the stored development bounds exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.intervals.freeze import reproduce_development
from gridcast.intervals.report import BASELINES, CONFORMAL_C, HEADLINE, find_runs
from tests.integration.test_m3_report import stored, world

pytestmark = pytest.mark.integration
__all__ = ["stored", "world"]  # fixtures


def test_reproduces_the_stored_development_bounds(world: Engine, stored: AppConfig) -> None:
    out = reproduce_development(world, stored, "VIC1", workers=2)
    assert [r.run for r in out] == [CONFORMAL_C, *BASELINES, HEADLINE]
    assert all(r.passed for r in out), [r for r in out if not r.passed]


def test_a_changed_stored_bound_is_caught(world: Engine, stored: AppConfig) -> None:
    run = find_runs(world, stored, "VIC1")[CONFORMAL_C]
    path = stored.data_root() / run.directory / f"{CONFORMAL_C}__hi80.parquet"
    frame = pd.read_parquet(path)
    values = frame["h003"].to_numpy(dtype="float64", copy=True)
    values[int(np.flatnonzero(np.isfinite(values))[0])] += 1.0
    frame["h003"] = values.astype("float32")
    frame.to_parquet(path, index=False)
    out = {r.run: r for r in reproduce_development(world, stored, "VIC1", workers=2)}
    assert not out[CONFORMAL_C].passed
    assert out[CONFORMAL_C].differing_bounds == (f"{CONFORMAL_C}__hi80.parquet",)
    assert out[CONFORMAL_C].selections_equal
    assert all(r.passed for name, r in out.items() if name != CONFORMAL_C)
