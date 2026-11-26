"""M5 step 5a: ``api-verify``'s comparisons, refusals, samples and P5 order, P6 against the app,
P3 on the synthetic walk-forward (with a tampered body), and ``report-m5``'s rendering."""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridcast.api import report as rep
from gridcast.api import verify as v
from gridcast.api.app import create_app
from gridcast.api.catalogue import read_only_engine
from gridcast.api.contract import INVALID_INPUTS
from gridcast.api.schema import SCHEMA_PATH
from gridcast.config import ServingVerifyConfig, load_config
from gridcast.serving.sources import SourceMatrix
from tests.unit.test_api_models import HOLDOUT, RUN, World, _client, _partial, build_world

H = 288
F32 = np.float32


def _ulp(x: float, toward: float = np.inf) -> float:
    return float(np.nextafter(F32(x), F32(toward)))


# --------------------------------------------------------------------------- comparisons


def test_float32_equality_is_exact_bits_and_nulls_where_nan() -> None:
    want = np.array([1.5, np.nan, 3.1415927, -0.0, 5000.25], dtype=F32)
    served: list[Any] = [1.5, None, float(F32(3.1415927)), -0.0, 5000.25]
    assert not v.float32_mismatches(served, want).any()
    cases = {
        0: _ulp(1.5),  # one ulp
        1: 0.0,  # a value where nothing was stored
        2: 3.1415927,  # the float64 decimal, not the float32 it names
        3: 0.0,  # +0 for -0 (different bits)
        4: None,  # null where a value was stored
    }
    for i, bad in cases.items():
        got = list(served)
        got[i] = bad
        assert v.float32_mismatches(got, want).tolist() == [k == i for k in range(5)], (i, bad)
    for bad in (True, "1.5", float("nan"), 1e300):
        assert v.float32_mismatches([bad], np.array([1.5], F32)).tolist() == [True], bad
    with pytest.raises(ValueError, match="served values"):
        v.float32_mismatches([1.5], want)


def test_database_values_compare_exactly() -> None:
    t = pd.Timestamp("2025-10-01 00:05", tz="UTC")
    assert v._equal("2025-10-01T00:05:00Z", t.to_pydatetime())
    assert v._equal("2025-10-01T10:05:00+10:00", t)
    assert not v._equal("2025-10-01T00:05:00", t)  # no offset
    assert not v._equal(1, True) and not v._equal(True, 1)
    assert v._equal(5123.4, 5123.4) and not v._equal(5123.4000001, 5123.4)
    assert v._equal(None, None) and not v._equal(0, None)
    assert v._equal(["a", "b"], ["a", "b"]) and not v._equal(["b", "a"], ["a", "b"])


def test_a_tally_counts_and_keeps_at_most_20_examples() -> None:
    t = v.Tally("P2", "x", requests=1)
    want = np.arange(30, dtype=F32)
    bad = t.numbers([float(x) + 1 for x in want], want, "ctx", [f"lead {i}" for i in range(30)])
    assert bad.all() and t.mismatches == 30 and t.values_compared == 30
    assert len(t.examples) == v.MAX_EXAMPLES and t.examples[0].startswith("ctx lead 0: served 1.0")
    assert not t.passed
    ok = v.Tally("P2", "y", requests=1)
    ok.check(True, "never")
    assert ok.passed and ok.fields_compared == 1
    assert not v.Tally("P2", "z").passed  # nothing was requested


# --------------------------------------------------------------------------- refusals


HEALTH = {
    "status": "ok",
    "build": {"git_sha": "a" * 40, "dirty": False, "source": "image"},
    "checks": {"database": {"ok": True, "detail": "reachable"}},
}


def _health(**build: Any) -> dict[str, Any]:
    return {**HEALTH, "build": {**HEALTH["build"], **build}}


@pytest.mark.parametrize(
    ("health", "checkout", "problem"),
    [
        (_health(), None, "commit is unknown"),
        (_health(), "a" * 40 + "-dirty", "uncommitted changes"),
        (_health(source="checkout"), "a" * 40, "not the container image"),
        (_health(dirty=True), "a" * 40, "not clean"),
        (_health(dirty=None), "a" * 40, "not clean"),
        (_health(git_sha="b" * 40), "a" * 40, "make api-up"),
        (
            {**HEALTH, "status": "degraded", "checks": {"model_files": {"ok": False}}},
            "a" * 40,
            "model_files",
        ),
        (None, "a" * 40, "no /health body"),
    ],
)
def test_a_run_is_refused_unless_the_image_is_the_clean_checkout(
    health: Any, checkout: str | None, problem: str
) -> None:
    problems = v.build_problems(health, checkout)
    assert any(problem in p for p in problems), problems


def test_a_clean_image_of_head_may_run() -> None:
    assert v.build_problems(_health(), "a" * 40) == []


def test_a_refused_run_measures_nothing() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, path: str, params: Any = None) -> httpx.Response:
            self.calls.append(path)
            return httpx.Response(200, json=_health(dirty=True))

    c = Client()
    with pytest.raises(v.VerifyRefused, match="not clean"):
        v.run_verify(c, None, load_config(), "a" * 40, "x", {}, truth=None)  # type: ignore[arg-type]
    assert c.calls == ["/health"]


# --------------------------------------------------------------------------- samples, P5


def _truth() -> v.Truth:
    """Two splits of 30-minute issue times; some rows empty, some partly stored."""
    rng = np.random.default_rng(3)
    hold = pd.Timestamp("2025-09-30 14:00", tz="UTC")
    times = {
        "dev": pd.date_range(hold - pd.Timedelta(days=10), hold, freq="30min", inclusive="left"),
        "holdout": pd.date_range(hold, periods=96, freq="30min"),
    }
    matrices, pubs = {}, {}
    for k, (split, ts) in enumerate(times.items()):
        point = (5000 + 100 * rng.standard_normal((len(ts), H))).astype(F32)
        point[::7] = np.nan  # empty issue times
        point[3, 100:] = np.nan
        matrices[("demand_point", split)] = SourceMatrix(ts, {"point": point}, 0)
        for product in (
            "demand_point",
            "demand_interval",
            "price_persistence",
            "price_median_profile",
        ):
            pubs[(product, split)] = v.Pub(10 * k + len(pubs), ts[0], ts[-1])
    folds = [
        v.Fold("dev", "r1", times["dev"][0], times["dev"][200]),
        v.Fold("dev", "r2", times["dev"][200], hold),
        v.Fold("holdout", "r3", hold, times["holdout"][-1] + pd.Timedelta(minutes=30)),
    ]
    days = list(pd.date_range("2021-10-01 14:00", periods=40, freq="D", tz="UTC"))
    events = [
        {
            "event_id": 100 + i,
            "first_interval_end": pd.Timestamp("2022-05-03", tz="UTC") + pd.Timedelta(days=9 * i),
        }
        for i in range(30)
    ]
    return v.Truth(
        region="VIC1",
        n_leads=H,
        step=5,
        issue_every=30,
        holdout_start=hold,
        pubs=pubs,
        matrices=matrices,
        folds=folds,
        anomalies=events,
        actual_days=days,
    )


VCFG = ServingVerifyConfig(
    seed=505,
    issue_times_per_fold=20,
    evolution_targets=50,
    actual_days=10,
    warmup_requests=3,
    stored_requests=40,
    model_requests=10,
    stored_p95_ms=100,
    model_p95_ms=500,
)


def test_samples_come_from_each_fold_among_stored_issue_times_and_are_seeded() -> None:
    truth = _truth()
    s = v.draw_samples(truth, VCFG)
    assert [len(ts) for _, ts in s.by_fold] == [20, 20, 20]
    for fold, ts in s.by_fold:
        stored = set(truth.stored_times(fold.split))
        assert ts == sorted(ts) and len(set(ts)) == len(ts)
        assert all(fold.start <= t < fold.end and t in stored for t in ts)
    assert s.candidates[0] == sum(1 for t in truth.stored_times("dev") if t < truth.folds[0].end)
    again = v.draw_samples(truth, VCFG)
    assert again.issue_times == s.issue_times and again.targets == s.targets
    other = v.draw_samples(truth, VCFG.model_copy(update={"seed": 506}))
    assert other.issue_times != s.issue_times
    first, last = truth.demand_span()
    assert all(first < t <= last + pd.Timedelta(hours=24) for t in s.targets)
    assert all((t - first) % pd.Timedelta(minutes=5) == pd.Timedelta(0) for t in s.targets)
    assert len(s.days) == 10 and set(s.days) <= set(truth.actual_days)
    small = v.draw_samples(truth, VCFG.model_copy(update={"issue_times_per_fold": 500}))
    assert [len(ts) for _, ts in small.by_fold] == small.candidates  # fewer: all of them
    json.dumps(s.as_dict())


def test_evolution_issues_are_the_published_ones_in_the_last_24_hours() -> None:
    truth = _truth()
    hold = truth.holdout_start
    issues = truth.evolution_issues(hold + pd.Timedelta(minutes=5))
    assert (
        len(issues) == 48
        and issues[-1] == hold
        and issues[0] == hold - pd.Timedelta(hours=23, minutes=30)
    )
    first, _ = truth.demand_span()
    assert truth.evolution_issues(first) == []
    assert truth.evolution_issues(first + pd.Timedelta(minutes=5)) == [first]


def test_p5_requests_are_seeded_and_interleaved_after_the_warmups() -> None:
    truth = _truth()
    endpoints = (*v.STORED_ENDPOINTS, *v.MODEL_ENDPOINTS)
    reqs = v.p5_requests(truth, VCFG, endpoints)
    for name, rs in reqs.items():
        want = VCFG.warmup_requests + (
            VCFG.model_requests if name in v.MODEL_ENDPOINTS else VCFG.stored_requests
        )
        assert len(rs) == want and {r.endpoint for r in rs} == {name}
    warm, timed = v.p5_order(reqs, VCFG.warmup_requests, VCFG.seed)
    assert len(warm) == 3 * len(endpoints)
    assert sorted(map(repr, warm)) == sorted(repr(r) for rs in reqs.values() for r in rs[:3])
    assert len(timed) == 6 * 40 + 2 * 10
    blocks = sum(a.endpoint != b.endpoint for a, b in pairwise(timed))
    assert blocks > 100  # interleaved, not one block per endpoint
    assert v.p5_order(v.p5_requests(truth, VCFG, endpoints), 3, 505) == (warm, timed)
    stored = set(truth.stored_times("dev").append(truth.stored_times("holdout")))
    model_times = {
        pd.Timestamp(r.params["issue_time"]) for r in timed if r.endpoint in v.MODEL_ENDPOINTS
    }
    assert model_times <= stored
    months = [r.params for r in timed if r.endpoint == "anomalies"]
    assert all(m["limit"] == "100" and pd.Timestamp(m["start"]).hour == 14 for m in months)


def test_latency_verdicts_use_p95_and_every_status() -> None:
    ms = [10.0] * 95 + [200.0] * 5
    ok = v.latency_stats(ms, [200] * 100, 100)
    assert ok["passed"] and ok["p50_ms"] == 10 and ok["p95_ms"] == pytest.approx(19.5)
    assert not v.latency_stats([10.0] * 94 + [200.0] * 6, [200] * 100, 100)["passed"]
    assert not v.latency_stats([10.0] * 100, [200] * 99 + [503], 100)["passed"]


def test_run_p5_times_every_request_and_keeps_the_log() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503 if "explain" in request.url.path else 200, json={"ok": 1})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")
    reqs = [v.Request("forecast", v.DEMAND, {"issue_time": "t"}) for _ in range(5)]
    bad = [v.Request("explain", v.EXPLAIN, {"issue_time": "t"}) for _ in range(5)]
    out = v.run_p5(client, reqs[:1], reqs + bad, {"forecast": 100, "explain": 500, "price": 100})
    assert set(out["endpoints"]) == {"forecast", "explain"}  # price was not requested
    assert out["endpoints"]["forecast"]["passed"] and out["endpoints"]["forecast"]["n"] == 5
    assert not out["endpoints"]["explain"]["passed"] and out["endpoints"]["explain"]["non_200"] == 5
    assert not out["passed"] and len(out["timed"]) == 10 and len(out["warmup"]) == 1
    json.dumps(out)


# --------------------------------------------------------------------------- P6 on the app


@pytest.fixture
def app_client() -> TestClient:
    engine = read_only_engine("postgresql+psycopg://nobody:x@127.0.0.1:9/none")
    return TestClient(create_app(engine=engine), raise_server_exceptions=False)


def test_the_contract_checks_pass_on_the_app(app_client: TestClient) -> None:
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema = v.check_schema(app_client, committed)  # type: ignore[arg-type]
    assert schema.passed, schema.examples
    assert schema.extra["paths"] == len(committed["paths"]) >= 10
    invalid = v.check_errors(app_client, "invalid", INVALID_INPUTS, 422)  # type: ignore[arg-type]
    assert invalid.passed and invalid.requests == len(INVALID_INPUTS), invalid.examples


def test_the_contract_checks_catch_drift_and_a_wrong_message(app_client: TestClient) -> None:
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    changed = {**committed, "info": {**committed["info"], "version": "0"}}
    s = v.check_schema(app_client, changed)  # type: ignore[arg-type]
    assert not s.passed and "['info']" in s.examples[0]
    table = [("/v1/forecasts/demand", {"issue_time": "noon"}, "no such text")]
    assert not v.check_errors(app_client, "x", table, 422).passed  # type: ignore[arg-type]
    table = [("/v1/forecasts/demand", {"issue_time": "noon"}, "a case")]
    assert v.check_errors(app_client, "x", table, 404).bad_responses == 1  # type: ignore[arg-type]


# --------------------------------------------------------------------------- P3 on the world


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return build_world(tmp_path_factory.mktemp("data"))


def _world_truth(w: World) -> v.Truth:
    matrices, pubs = {}, {}
    for split in ("dev", "holdout"):
        ts = w.split_times(split)
        rows = w.stored[[w.row(t) for t in ts]]
        matrices[("demand_point", split)] = SourceMatrix(ts, {"point": rows.copy()}, 0)
        pubs[("demand_point", split)] = v.Pub(RUN[split], ts[0], ts[-1])
    return v.Truth("VIC1", H, 5, 30, HOLDOUT, pubs, matrices)


def _times(w: World) -> list[pd.Timestamp]:
    full = [t for t in w.split_times("dev") if np.isfinite(w.stored[w.row(t)]).all()]
    return [full[0], _partial(w, "dev"), _partial(w, "holdout"), w.split_times("holdout")[-1]]


def test_p3_passes_on_the_synthetic_walk_forward(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(world, monkeypatch)
    truth, times = _world_truth(world), _times(world)
    tol = world.cfg.anomalies.explain.sum_tolerance
    rc = v.check_recompute(client, truth, times)
    assert rc.passed, rc.examples
    assert rc.extra == {"pairs": 4 * H, "pairs_equal": 4 * H} and rc.values_compared == 2 * 4 * H
    ex = v.check_explain(client, truth, times, tol)
    assert ex.passed, ex.examples
    stored = sum(int(np.isfinite(truth.point(t)).sum()) for t in times)
    assert ex.extra == {"issue_times": 4, "explained": 4, "leads": stored}


def test_p3_catches_a_one_ulp_difference_against_the_source(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(world, monkeypatch)
    truth, times = _world_truth(world), _times(world)
    t = times[1]
    m = truth.matrices[("demand_point", "dev")]
    i = int(m.times.get_indexer([t])[0])
    h = int(np.flatnonzero(np.isfinite(m.by_column["point"][i]))[0])
    m.by_column["point"][i, h] = np.nextafter(m.by_column["point"][i, h], F32(np.inf))
    rc = v.check_recompute(client, truth, times)
    assert not rc.passed and rc.mismatches == 2  # recomputed and stored, one lead
    assert rc.extra["pairs_equal"] == 4 * H - 1
    ex = v.check_explain(client, truth, times, world.cfg.anomalies.explain.sum_tolerance)
    assert not ex.passed and ex.extra["explained"] == 3
    assert any(f"lead {h + 1}" in e for e in ex.examples)


class _Tamper:
    """The app, with one contribution of every explain body changed."""

    def __init__(self, inner: TestClient, delta: float) -> None:
        self.inner, self.delta = inner, delta

    def get(self, path: str, params: Any = None) -> httpx.Response:
        r = self.inner.get(path, params=params)
        if not path.endswith("/explain") or r.status_code != 200:
            return r
        body = r.json()
        body["leads"][0]["features"][0]["contribution_mw"] += self.delta
        return httpx.Response(200, json=body)


def test_p3_re_adds_each_decomposition(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    truth, times = _world_truth(world), _times(world)[:1]
    tol = world.cfg.anomalies.explain.sum_tolerance
    client = _Tamper(_client(world, monkeypatch), 1e-3)
    ex = v.check_explain(client, truth, times, tol)  # type: ignore[arg-type]
    assert not ex.passed and ex.mismatches == 2  # the sum, and families against features
    assert "anchor + base + Σ contributions" in ex.examples[0]
    tiny = _Tamper(_client(world, monkeypatch), tol / 100)
    assert v.check_explain(tiny, truth, times, tol).passed  # type: ignore[arg-type]


# --------------------------------------------------------------------------- report-m5


def _run(sha: str, started: str, explain_p95: float = 300.0) -> dict[str, Any]:
    def tally(criterion: str, name: str, **extra: Any) -> v.Tally:
        return v.Tally(
            criterion, name, requests=280, values_compared=10, fields_compared=5, extra=extra
        )

    checks = [
        tally("P6", "schema", paths=11),
        tally("P6", "invalid inputs (422)", cases=19),
        tally("P6", "nothing published (404)", cases=12),
        tally("P6", "no-500 search", statuses={"200": 10, "422": 290}, examples=300),
        tally("P2", "demand forecasts"),
        tally("P2", "price baselines", not_published=1),
        tally("P2", "forecast evolution", items=13_000),
        tally("P2", "actuals", intervals=14_400),
        tally("P2", "anomaly events", pages=12, events=1127, limit=100),
        tally("P3", "recompute", pairs=80_640, pairs_equal=80_640),
        tally("P3", "explain", issue_times=280, explained=280, leads=70_000),
    ]
    limits = {"forecast": 100.0, "explain": 500.0}
    p5 = {
        "endpoints": {
            name: {
                **v.latency_stats(
                    [10.0, explain_p95 if name == "explain" else 20.0], [200, 200], lim
                ),
                "median_kib": 9.0,
            }
            for name, lim in limits.items()
        }
    }
    p5["passed"] = all(e["passed"] for e in p5["endpoints"].values())
    result = v.VerifyResult(
        url="http://x",
        started_at=started,
        checkout_git_sha=sha,
        build={"git_sha": sha, "dirty": False, "source": "image"},
        settings={**VCFG.model_dump(), "sum_tolerance": 1e-6, "max_examples": 20},
        environment={
            "vm": {
                "NCPU": 4,
                "MemTotal": 4 * 2**30,
                "OperatingSystem": "Docker Desktop",
                "Architecture": "aarch64",
                "ServerVersion": "29",
            }
        },
        samples={
            "folds": [
                {
                    "split": "dev",
                    "retrain": "2023-08-01",
                    "start": "a",
                    "end": "b",
                    "candidates": 4000,
                    "issue_times": ["x"] * 20,
                }
            ],
            "evolution_targets": ["t"] * 280,
            "actual_days": ["d"] * 50,
        },
        checks=checks,
        p5=p5,
        reported={
            "memory_after": {"gridcast-api": {"mem_usage": "800MiB / 3.8GiB", "mem_perc": "20%"}},
            "point_outside_50": {
                "dev": {"pairs": 10, "below": 2, "above": 3, "outside": 5, "share": 0.5}
            },
        },
    )
    result.build_after = result.build
    return result.as_dict()


def _pubs() -> list[dict[str, Any]]:
    out = []
    for product in ("demand_point", "demand_interval", "price_persistence", "price_median_profile"):
        for split in ("dev", "holdout"):
            checks: dict[str, Any] = {
                "p1": {"rows_read": 2 * H - 3, "equal": 2 * H - 3, "complete": True, "exact": True}
            }
            if product == "demand_interval":
                checks["sql"] = {"p4": {"pairs": 2 * H - 3, "non_nested": 0}}
                checks["interval_metrics"] = {"db_equals_parquet": True}
            out.append(
                {
                    "product": product,
                    "split": split,
                    "run_id": 1,
                    "source_run_id": 1,
                    "first_issue": pd.Timestamp("2023-07-31 14:00", tz="UTC"),
                    "last_issue": pd.Timestamp("2023-07-31 14:30", tz="UTC"),
                    "issue_times": 2,
                    "rows_expected": 2 * H - 3,
                    "missing_pairs": 3,
                    "git_sha": "abc1234",
                    "status": "success",
                    "checks": checks,
                }
            )
    return out


def test_the_report_leads_with_the_first_run_and_lists_later_runs() -> None:
    cfg = load_config()
    first = _run("a" * 40, "2026-11-28T01:00:00Z", explain_p95=900.0)
    second = _run("b" * 40, "2026-11-28T03:00:00Z")
    smokes = [
        {
            "started_at": "2026-11-25T14:29:01Z",
            "build": {"git_sha": "c" * 40},
            "passed": True,
            "checks": [
                {"name": "explain", "target": "dev 2024-03-15T02:00:00Z", "ms": [341.2, 337.0]}
            ],
        }
    ]
    log = {("a" * 40, "b" * 40): ["bbbbbbb Speed up explain"]}
    text_ = rep.render([first, second], _pubs(), smokes, cfg, lambda a, b: log[(a, b)])
    assert "**5 of 6 criteria pass** on the first verification run (2026-11-28T01:00:00Z" in text_
    assert "explain 855.5 ms (FAIL)" in text_  # the first run's p95 of (10, 900), linear
    assert (
        "| 2 | 2026-11-28T03:00:00Z | `bbbbbbbbbbbb` | `bbbbbbb` Speed up explain | "
        "pass | pass | pass | pass |" in text_
    )
    assert (
        "| 1 | 2026-11-28T01:00:00Z | `aaaaaaaaaaaa` | the first run | "
        "pass | pass | **fail** | pass |" in text_
    )
    assert f"{2 * (2 * H - 3) * 4:,} rows read back over 8 of 8 publications" in text_
    assert "0 of 1,146 published interval pairs are not nested" in text_
    assert "341, 337" in text_ and "50.0%" in text_ and "4 CPUs, 4.0 GiB" in text_
    for d in rep.DISCLOSURES:
        assert d.text in text_


def test_the_report_fails_p1_on_a_count_or_value_problem() -> None:
    pubs = _pubs()
    pubs[0]["missing_pairs"] = 4
    pubs[3]["checks"]["interval_metrics"]["db_equals_parquet"] = False
    rows = rep.p1_p4(pubs, H)
    ok, body = rows["P1"]
    assert not ok and "demand_point/dev: rows" in body and "hit counts differ" in body
    pubs = [p for p in _pubs() if (p["product"], p["split"]) != ("demand_interval", "holdout")]
    assert not rep.p1_p4(pubs, H)["P1"][0] and not rep.p1_p4(pubs, H)["P4"][0]
    pubs = _pubs()
    pubs[2]["checks"]["sql"]["p4"]["non_nested"] = 1
    assert rep.p1_p4(pubs, H)["P1"][0] and not rep.p1_p4(pubs, H)["P4"][0]
    with pytest.raises(ValueError, match="no verification run"):
        rep.render([], _pubs(), [], load_config())
