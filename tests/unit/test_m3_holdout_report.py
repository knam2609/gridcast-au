"""U5 from what is stored, and the holdout's selection filter (M3 step 6c)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gridcast.intervals.holdout_report import HoldoutRuns, holdout_selections, u5_criterion
from gridcast.intervals.report import HEADLINE, IntervalRun

RETRAINS = ["2025-07-31 14:00:00+00:00", "2025-10-31 13:00:00+00:00"]


def _run(name: str, sha: str = "abc1234", passed: bool = True, **extra: Any) -> IntervalRun:
    config = {
        "code_sha": sha,
        "reproduction": {"passed": passed, "development_runs": [{}, {}]},
        "holdout_retrains": RETRAINS,
        **extra,
    }
    return IntervalRun(1, "b1", name, sha, Path("x"), config)


def _runs(batches: list[str], **kw: Any) -> HoldoutRuns:
    five = {HEADLINE: _run(HEADLINE, **kw), "conformal_lgbm_C": _run("conformal_lgbm_C")}
    return HoldoutRuns("b1", batches, five, {"cqr30_lgbm_C": _run("cqr30_lgbm_C")}, _run("qr"))


def test_u5_passes_for_one_reproduced_batch_from_one_commit() -> None:
    c = u5_criterion(_runs(["b1"]), "hash")
    assert c.key == "U5" and c.passed
    assert "1 holdout interval batch(es)" in c.measured and "reproduced the 2" in c.measured


def test_u5_fails_for_a_second_batch_a_second_commit_or_no_reproduction() -> None:
    assert not u5_criterion(_runs(["b1", "b2"]), "hash").passed
    assert not u5_criterion(_runs(["b1"], sha="def5678"), "hash").passed  # two commits
    failed = u5_criterion(_runs(["b1"], passed=False), "hash")
    assert not failed.passed and "did NOT reproduce" in failed.measured


def test_holdout_selections_keep_only_the_retrains_applying_to_the_holdout() -> None:
    selections = [
        {"retrain": "2025-04-30T14:00:00+00:00", "level": 80, "band": "h001-006"},
        {"retrain": "2025-07-31T14:00:00+00:00", "level": 80, "band": "h001-006"},
        {"retrain": "2025-10-31T13:00:00+00:00", "level": 80, "band": "h001-006"},
    ]
    run = holdout_selections(_run(HEADLINE, selections=selections))
    assert [s["retrain"][:10] for s in run.config["selections"]] == ["2025-07-31", "2025-10-31"]
    assert holdout_selections(_run("qr_lgbm_C")).config.get("selections") is None  # raw QR
