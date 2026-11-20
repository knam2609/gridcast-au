"""Pure parts of the two step-6 reports (m4_plan.md §1, "Clarified in step 6")."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.anomalies.holdout_report import (
    E1_GLOBAL,
    E1_PEAKS,
    Row,
    holdout_cell,
    replicated_holds,
    verdict_table,
)
from gridcast.config import load_config
from gridcast.evaluation.m2_report import Criterion
from gridcast.explainability.associations import AssociationData, development_binnings
from gridcast.explainability.holdout_report import (
    EdgesMismatch,
    check_binnings,
    stored_binnings,
)


def _c(key: str, passed: bool) -> Criterion:
    return Criterion(key, f"rule {key}", f"measured {key}", passed)


def _rows(fail: set[str], fail_v2: set[str] | None = None) -> list[Row]:
    fail_v2 = fail_v2 or set()
    keys = ["A1", "A2", "A3", "A4", "A5", E1_PEAKS, E1_GLOBAL, "A6"]
    rows = []
    for k in keys:
        hold = {
            1: None if k == "A2" else _c(k, k not in fail),
            2: None if k == "A2" else _c(k, k not in fail_v2),
        }
        dev = {1: None if k == "A6" else _c(k, True), 2: None if k == "A6" else _c(k, True)}
        rows.append(Row(k, f"rule {k}", hold, dev))
    return rows


def test_replicated_criteria_count_e1_once_and_skip_a2_and_a6() -> None:
    assert replicated_holds(_rows(set()), 1) == (5, 5)
    assert replicated_holds(_rows({"A1", "A6"}), 1) == (4, 5)
    # E1 holds only if both its parts hold.
    assert replicated_holds(_rows({E1_GLOBAL}), 1) == (4, 5)
    assert replicated_holds(_rows({E1_PEAKS, E1_GLOBAL}), 1) == (4, 5)
    assert replicated_holds(_rows(set(), {"A3"}), 2) == (4, 5)


def test_holdout_cells_say_holds_and_a6_says_pass() -> None:
    assert holdout_cell(_c("A1", True), "A1").startswith("**holds**")
    assert holdout_cell(_c("A1", False), "A1").startswith("**does not hold**")
    assert holdout_cell(_c("A6", True), "A6").startswith("**PASS**")
    assert "no class-P event" in holdout_cell(None, "A2")
    table = verdict_table(_rows({"A3"}))
    assert "v2 holdout (blind)" in table and "v1 development (pre-registered)" in table
    assert table.count("\n") == 2 + 8 - 1


def _data(seed: int = 3) -> AssociationData:
    rng = np.random.default_rng(seed)
    n = 288 * 20
    ends = pd.date_range("2024-01-01 00:05", periods=n, freq="5min", tz="UTC")
    solar = np.where(rng.random(n) < 0.45, 0.0, rng.uniform(1, 2000, n))
    observed = {
        "demand": rng.normal(5000, 800, n),
        "temperature": rng.normal(15, 6, n),
        "wind": rng.uniform(0, 3000, n),
        "solar": solar,
        "net_interchange": rng.normal(0, 400, n),
        "hour": np.asarray(ends.hour, dtype=np.int64),
        "month": np.asarray(ends.month, dtype=np.int64),
    }
    day = np.repeat(np.arange(20), 288) + 700_000
    return AssociationData(ends, day, rng.normal(80, 50, n), observed)


def _write_edges(bins: dict, path: Path) -> None:
    """As step 5's report writes them."""
    edges = {k: {"kind": b.kind, "edges": list(b.edges), "width": b.width} for k, b in bins.items()}
    path.write_text(json.dumps({"period": ["a", "b"], **edges}, indent=1) + "\n")


def test_stored_edges_round_trip_to_the_development_bins(tmp_path: Path) -> None:
    cfg = load_config().explain_report.associations
    bins = development_binnings(_data(), cfg)
    _write_edges(bins, tmp_path / "edges.json")
    stored = stored_binnings(tmp_path / "edges.json")
    check_binnings(stored, bins)  # exact, categories included
    assert stored["hour"].categories == tuple(range(24))
    assert stored["demand_two_way"].variable == "demand"


def test_changed_edges_are_refused(tmp_path: Path) -> None:
    cfg = load_config().explain_report.associations
    _write_edges(development_binnings(_data(3), cfg), tmp_path / "edges.json")
    other = development_binnings(_data(4), cfg)
    try:
        check_binnings(stored_binnings(tmp_path / "edges.json"), other)
    except EdgesMismatch as exc:
        assert "demand" in str(exc)
    else:
        raise AssertionError("different edges were accepted")
