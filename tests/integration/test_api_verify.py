"""M5 step 5a on the test database: ``api-verify``'s checks against the published fixture.

The fixture has no stored fits, so recompute and explain answer 503 here (P3 is tested on
the synthetic walk-forward in the unit suite). Every stored-endpoint check must pass, catch a
one-ulp change in the database and an edited event, and the whole run must complete, save
and render.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.api import report as rep
from gridcast.api import verify as v
from gridcast.api.contract import INVALID_INPUTS, SEARCH_PARAMETERS
from gridcast.api.schema import SCHEMA_PATH
from tests.integration.test_api import Env, make_env
from tests.integration.test_publish import DEV_TIMES, HOLD_TIMES, H

pytestmark = pytest.mark.integration

HALF = pd.Timedelta(minutes=30)
MODEL_ROUTES = (v.RECOMPUTE, v.EXPLAIN)


@pytest.fixture
def fx_env(db: Engine, tmp_path: Path) -> Iterator[Env]:
    with make_env(db, tmp_path) as e:
        yield e


def _truth(env: Env) -> v.Truth:
    folds = [
        v.Fold("dev", "r1", DEV_TIMES[0], DEV_TIMES[48]),
        v.Fold("dev", "r2", DEV_TIMES[48], DEV_TIMES[-1] + HALF),
        v.Fold("holdout", "r3", HOLD_TIMES[0], HOLD_TIMES[-1] + HALF),
    ]
    return v.load_truth(env.db, env.fx.cfg, folds=folds)


def _targets(truth: v.Truth) -> list[pd.Timestamp]:
    five = pd.Timedelta(minutes=5)
    hold = truth.holdout_start
    return [
        DEV_TIMES[0] + five,  # one issue time before it
        DEV_TIMES[30] + 37 * five,
        DEV_TIMES[-1] + 288 * five,  # the last development lead
        DEV_TIMES[-1] + 289 * five,  # nothing published targets it: 404
        hold + five,  # the holdout's first issue time only (this fixture's dev is in 2023)
        HOLD_TIMES[-1] + 12 * five,
    ]


def test_the_stored_endpoint_checks_pass_on_the_published_fixture(fx_env: Env) -> None:
    env = fx_env
    truth = _truth(env)
    assert truth.other_event == env.v1_event
    assert {r["event_id"] for r in truth.anomalies} == set(env.events)
    times = list(DEV_TIMES) + list(HOLD_TIMES)
    client = env.client

    demand = v.check_demand(client, truth, times)
    assert demand.passed, demand.examples
    stored = sum(int(np.isfinite(truth.point(t)).sum()) for t in times)
    assert demand.values_compared == 7 * H * len(times)  # missing pairs too: null for NaN
    assert stored < H * len(times)

    price = v.check_price(client, truth, times)
    assert price.passed, price.examples
    assert price.extra["not_published"] == len(DEV_TIMES)  # the price file starts in 2025

    targets = _targets(truth)
    assert [len(truth.evolution_issues(t)) for t in targets] == [1, 37, 1, 0, 1, 47]
    evo = v.check_evolution(client, truth, targets)
    assert evo.passed, evo.examples
    assert evo.extra["items"] == 87 and evo.requests == len(targets)

    days = truth.actual_days[:3]
    actuals = v.check_actuals(client, truth, days)
    assert actuals.passed and actuals.extra["intervals"] == 3 * H, actuals.examples

    events = v.check_anomalies(client, truth, limit=3)
    assert events.passed, events.examples
    assert events.extra["pages"] == 3 and events.extra["events"] == len(env.events) == 8

    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert v.check_schema(client, committed).passed
    assert v.check_errors(client, "422", INVALID_INPUTS, 422).passed
    table = [r for r in v.not_published_table(truth) if r[0] not in MODEL_ROUTES]
    assert {r[2] for r in table} >= {"an unpublished region", "an event that is not served (v1)"}
    nf = v.check_errors(client, "404", table, 404)
    assert nf.passed, nf.examples
    stored_params = [p for p in SEARCH_PARAMETERS if p[0] not in MODEL_ROUTES]
    search = v.run_search(client, v.search_base(truth, times[0]), stored_params, examples=60)
    assert search.passed and search.requests >= 50, search.examples

    out = v.point_outside_50(env.db, truth)
    assert out["dev"]["pairs"] == int(np.isfinite(env.fx.point["dev"]).sum())
    assert out["holdout"]["pairs"] == H * len(HOLD_TIMES)


def test_the_checks_catch_a_one_ulp_change_and_an_edited_event(fx_env: Env) -> None:
    env = fx_env
    truth = _truth(env)
    t = DEV_TIMES[5]
    run = truth.pubs[("demand_point", "dev")].run_id
    with env.db.begin() as conn:
        conn.execute(
            text(
                "UPDATE forecasts SET point = (point::real + (abs(point) * 1e-7)::real)::float8 "
                "WHERE run_id = :r AND issue_time = :t AND horizon = 17"
            ),
            {"r": run, "t": t.to_pydatetime()},
        )
        event = truth.anomalies[2]["event_id"]
        conn.execute(
            text("UPDATE anomalies SET summary = summary || '.' WHERE event_id = :e"), {"e": event}
        )
    demand = v.check_demand(env.client, truth, [t, DEV_TIMES[6]])
    assert demand.mismatches == 1 and "lead 17" in demand.examples[0]
    events = v.check_anomalies(env.client, truth, limit=3)
    assert events.mismatches == 2  # in its page and by id
    assert all("summary" in e for e in events.examples)


def test_a_whole_run_completes_saves_and_renders(
    fx_env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recompute and explain have no fits here (503), so P3 fails, and P5 and P6 with it;
    P2 passes. The point is that the run completes, is saved once, and renders."""
    env = fx_env
    cfg = env.fx.cfg
    cfg.serving.verify = cfg.serving.verify.model_copy(
        update={"stored_requests": 15, "model_requests": 5, "warmup_requests": 2}
    )
    monkeypatch.setattr(v, "build_problems", lambda health, checkout: [])
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    truth = _truth(env)
    result = v.run_verify(
        env.client, env.db, cfg, "f" * 40, "testclient", committed, truth=truth, docker=False
    )  # type: ignore[arg-type]
    verdicts = result.verdicts()
    assert verdicts["P2"] and not verdicts["P3"] and not result.passed
    p3 = [c for c in result.checks if c.criterion == "P3"]
    assert all(c.bad_responses == c.requests > 0 for c in p3)  # 503: no fits
    assert result.p5["endpoints"]["forecast"]["non_200"] == 0
    assert result.p5["endpoints"]["explain"]["non_200"] == 5 + 0  # timed only
    path = v.save(result, cfg)
    assert path.parent == cfg.data_root() / "serving"
    with pytest.raises(FileExistsError):
        v.save(result, cfg)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["verdicts"] == verdicts and len(stored["p5"]["timed"]) == 6 * 15 + 2 * 5
    pubs = rep.load_publications(env.db, "VIC1")
    text_ = rep.render(rep.load_runs(path.parent), pubs, [], cfg, lambda a, b: [])
    assert "| P1 |" in text_ and "**PASS**: " in text_ and "**FAIL**: " in text_
    assert v.render(result).startswith("api-verify testclient at ")
