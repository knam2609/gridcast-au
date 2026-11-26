"""M5 step 5a on the host: ``api-verify``'s truth and samples on the main data (read-only).

Runs only with ``make test-host``. It reads the main database and the parquet sources under
the data root and sends no request to the API, so it measures nothing: P2, P3, P5 and P6 are
judged by ``gridcast api-verify`` against the container (m5_plan.md §3).
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from gridcast.api import verify as v
from gridcast.api.catalogue import read_only_engine
from gridcast.config import SERVING_SPLITS, get_settings, load_config

pytestmark = pytest.mark.host

HALF = pd.Timedelta(minutes=30)


@pytest.fixture(scope="module")
def db() -> Iterator[Engine]:
    engine = read_only_engine(get_settings().database_url)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def truth(db: Engine) -> v.Truth:
    cfg = load_config()
    if not cfg.resolve(cfg.paths.data_root).is_dir():
        pytest.skip("the data root is not mounted")
    return v.load_truth(db, cfg)


def test_the_folds_are_the_published_fits_retrains(truth: v.Truth) -> None:
    by = {s: [f for f in truth.folds if f.split == s] for s in SERVING_SPLITS}
    assert [f.retrain for f in by["dev"]] == [
        "2023-08-01", "2023-11-01", "2024-02-01", "2024-05-01", "2024-08-01",
        "2024-11-01", "2025-02-01", "2025-05-01", "2025-08-01",
    ]  # fmt: skip
    assert [f.retrain for f in by["holdout"]] == [
        "2025-08-01", "2025-11-01", "2026-02-01", "2026-05-01", "2026-08-01",
    ]  # fmt: skip
    for split, folds in by.items():
        pub = truth.pubs[("demand_point", split)]
        assert folds[0].start == pub.first and folds[-1].end == pub.last + HALF
        assert all(a.end == b.start for a, b in pairwise(folds))
    assert by["dev"][-1].end == by["holdout"][0].start == truth.holdout_start


def test_the_truth_matrices_are_the_published_issue_times(truth: v.Truth) -> None:
    for (product, split), m in truth.matrices.items():
        pub = truth.pubs[(product, split)]
        assert m.times[0] == pub.first and m.times[-1] == pub.last, (product, split)
        assert m.grid_gaps == 0
    assert len(truth.anomalies) == 994 + 133
    assert truth.other_event is not None and truth.actual_days
    assert truth.actuals is not None
    rows = truth.actuals(truth.actual_days[0], truth.actual_days[0] + pd.Timedelta(days=1))
    assert len(rows) == 288


def test_the_samples_and_p5_requests_have_the_planned_sizes(truth: v.Truth) -> None:
    vcfg = load_config().serving.verify
    s = v.draw_samples(truth, vcfg)
    assert [len(ts) for _, ts in s.by_fold] == [vcfg.issue_times_per_fold] * 14
    assert len(s.issue_times) == 280 and len(s.targets) == 280 and len(s.days) == 50
    for t in s.issue_times:
        assert np.isfinite(truth.point(t)).any()
    assert any(not np.isfinite(truth.point(t)).all() for t in s.issue_times)  # some gaps
    reqs = v.p5_requests(truth, vcfg, (*v.STORED_ENDPOINTS, *v.MODEL_ENDPOINTS))
    warm, timed = v.p5_order(reqs, vcfg.warmup_requests, vcfg.seed)
    assert len(warm) == 8 * 20 and len(timed) == 6 * 500 + 2 * 200 == 3400
    table = v.not_published_table(truth)
    cases = {r[2] for r in table}
    assert "explain where nothing is stored (dev)" in cases and "before actuals" in cases
