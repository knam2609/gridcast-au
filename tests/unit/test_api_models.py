"""M5 step 3: recompute and explain on a small synthetic M2 walk-forward, without a database.

One LightGBM C spec searches every target form, so the fits mix forms. Its first retrain
plays the development batch and its second the holdout batch; the synthetic weather's gaps
leave some leads without a stored forecast, as run 2's missing weather did. The catalogue
is built by hand and the stored forecasts come from the walk-forward (``queries.points`` is
patched), so every request goes through the app.
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridcast.api import queries as q
from gridcast.api import runtime as rt
from gridcast.api import smoke
from gridcast.api.app import create_app
from gridcast.api.catalogue import Catalogue, Published, RunInfo, read_only_engine
from gridcast.config import AppConfig, LightGBMConfig, load_config
from gridcast.explainability.global_run import ANCHOR_TERMS
from gridcast.explainability.pairs import ModelCache
from gridcast.explainability.sample import feature_rows, origin_positions
from gridcast.explainability.stored import fits_from_records
from gridcast.features.builder import FeatureBuilder
from gridcast.forecasting import walkforward as wf
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
from gridcast.forecasting.origins import origin_positions as op
from gridcast.models.baselines import DAY_STEPS
from gridcast.models.ml import TARGET_FORMS
from tests.feature_fixtures import hourly_weather, weather_inputs
from tests.forecast_fixtures import make_ctx

SETUP = ForecastSetup()
H = SETUP.horizon_steps
RETRAINS = ("2024-04-01", "2024-04-15")  # development fold, then the holdout fold
EVAL_END = "2024-04-22"
HOLDOUT = nem_midnight_utc(RETRAINS[1])
RUN = {"dev": 39, "holdout": 61}
DEMAND = "/v1/forecasts/demand"


def _cfg(root: Path) -> AppConfig:
    base = load_config()
    ml = base.ml.model_copy(
        update={
            "early_stopping_weeks": 1,
            "lightgbm": LightGBMConfig(
                learning_rate=0.2,
                min_data_in_leaf=50,
                max_rounds=25,
                early_stopping_rounds=5,
                num_threads=2,
            ),
        }
    )
    cfg = base.model_copy(update={"ml": ml})
    cfg.paths.data_root = root
    return cfg


def _demand(pos: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(11)
    return 5000 + 800 * np.sin(2 * np.pi * pos / DAY_STEPS) + rng.normal(0, 120, len(pos))


@cache
def _builder() -> FeatureBuilder:
    ctx = make_ctx(start="2024-03-01 00:05", days=60, values=_demand)
    fc, obs = hourly_weather(ctx.times, seed=5)
    return FeatureBuilder(ctx, SETUP, weather_inputs(ctx.times, fc, obs))


@dataclasses.dataclass
class World:
    root: Path
    cfg: AppConfig
    features: rt.Features
    times: pd.DatetimeIndex  # every issue time forecast
    stored: np.ndarray  # (n, H) float32, NaN where nothing was stored

    def row(self, t: pd.Timestamp) -> int:
        pos = int(self.times.get_indexer(pd.DatetimeIndex([t]))[0])
        assert pos >= 0, t
        return pos

    def points(self, t: pd.Timestamp) -> dict[int, float]:
        v = self.stored[self.row(t)]
        return {h: float(v[h - 1]) for h in range(1, H + 1) if np.isfinite(v[h - 1])}

    def split_times(self, split: str) -> pd.DatetimeIndex:
        return self.times[(self.times < HOLDOUT) == (split == "dev")]


def build_world(root: Path) -> World:
    """The synthetic walk-forward under ``root`` (also used by the api-verify tests)."""
    cfg = _cfg(root)
    builder = _builder()
    retrains = [nem_midnight_utc(d) for d in RETRAINS]
    origins = op(builder.ctx, SETUP, retrains[0])
    origins = origins[builder.ctx.times[origins] < nem_midnight_utc(EVAL_END)]
    candidates = tuple({"target_form": f, "objective": "l1", "num_leaves": 7} for f in TARGET_FORMS)
    spec = wf.ModelSpec("lgbm_C", "C", "lightgbm", candidates=candidates)
    result = wf.run_walk_forward(
        builder, [spec], origins, retrains, cfg.ml, model_dir=root / "models", data_root=root
    )
    records = [dataclasses.asdict(r) for r in result.retrains["lgbm_C"]]
    end = nem_midnight_utc(EVAL_END)
    fits = {
        "dev": fits_from_records(
            [r for r in records if r["retrain"] == RETRAINS[0]],
            root,
            cfg.ml.lead_bands,
            (retrains[0], HOLDOUT),
        ),
        "holdout": fits_from_records(
            [r for r in records if r["retrain"] == RETRAINS[1]],
            root,
            cfg.ml.lead_bands,
            (HOLDOUT, end),
        ),
    }
    times = pd.DatetimeIndex(builder.ctx.times[origins]).as_unit("us")
    features = rt.Features(builder, fits, 0.0)
    return World(root, cfg, features, times, result.predictions["lgbm_C"])


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return build_world(tmp_path_factory.mktemp("data"))


def _catalogue(w: World, dev_batch: str | None = None) -> Catalogue:
    ex = w.cfg.anomalies.explain
    batches = {"dev": dev_batch or ex.dev_batch, "holdout": ex.holdout_batch}
    published, runs = {}, {}
    for k, split in enumerate(("dev", "holdout")):
        st = w.split_times(split)
        published[("demand_point", split)] = Published(
            "demand_point", split, RUN[split], RUN[split], st[0], st[-1], len(st), 0, 0, k, None
        )
        runs[RUN[split]] = RunInfo(RUN[split], batches[split], "lgbm_C", "ml", "success", None, {})
    return Catalogue(
        region="VIC1",
        first_issue=w.times[0],
        holdout_start=HOLDOUT,
        holdout_end=nem_midnight_utc(EVAL_END),
        published=published,
        runs=runs,
        metrics={},
        anomaly_runs={},
    )


def _client(
    w: World, monkeypatch: pytest.MonkeyPatch, stored: dict[int, float] | None = None, **kw: str
) -> TestClient:
    """The app on the world, without its lifespan (the runtime loads on first use)."""
    engine = read_only_engine("postgresql+psycopg://nobody:x@127.0.0.1:9/none")
    runtime = rt.ModelRuntime(w.cfg, None, features=w.features)
    app = create_app(w.cfg, engine, runtime)
    app.state.api._catalogue = _catalogue(w, **kw)

    def points(engine: object, run_id: int, t: pd.Timestamp) -> dict[int, float]:
        assert run_id == RUN["dev" if t < HOLDOUT else "holdout"]
        return dict(stored) if stored is not None else w.points(t)

    monkeypatch.setattr(q, "points", points)
    return TestClient(app, raise_server_exceptions=False)


def _partial(w: World, split: str) -> pd.Timestamp:
    """An issue time of ``split`` with some, not all, leads stored."""
    for t in w.split_times(split):
        n = np.isfinite(w.stored[w.row(t)]).sum()
        if 0 < n < H:
            return t
    raise AssertionError(f"no partly stored {split} issue time in the fixture")


def _iso(t: pd.Timestamp) -> str:
    return t.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------- recompute


@pytest.mark.parametrize("split", ["dev", "holdout"])
def test_recompute_reproduces_the_stored_forecast_and_its_gaps(
    world: World, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    t = _partial(world, split)
    r = _client(world, monkeypatch).get(f"{DEMAND}/recompute", params={"issue_time": _iso(t)})
    assert r.status_code == 200, r.text
    b = r.json()
    stored = world.stored[world.row(t)]
    assert b["split"] == split and b["point_run_id"] == RUN[split]
    assert b["matches_stored"] and b["leads_differing"] == 0 and b["max_abs_diff_mw"] == 0
    assert b["leads_stored"] == b["leads_recomputed"] == int(np.isfinite(stored).sum())
    assert 0 < b["max_float64_diff_mw"] < 0.01  # float64 vs its float32 cast
    got = np.array(
        [np.nan if x["recomputed_mw"] is None else x["recomputed_mw"] for x in b["leads"]]
    )
    np.testing.assert_array_equal(got.astype(np.float32), stored)  # NaN where nothing stored
    assert all(x["matches"] for x in b["leads"])
    assert all((x["fit"] is None) == (x["recomputed_mw"] is None) for x in b["leads"])
    fits = b["fits"]
    assert [(f["first_lead"], f["last_lead"]) for f in fits] == [
        tuple(band) for band in world.cfg.ml.lead_bands
    ]
    assert {f["retrain"] for f in fits} == {RETRAINS[0] if split == "dev" else RETRAINS[1]}
    assert all(f["model_file"].startswith("models/") for f in fits)
    assert sum(f["leads_recomputed"] for f in fits) == b["leads_recomputed"]
    batch = world.cfg.anomalies.explain.dev_batch if split == "dev" else None
    assert all(f["batch_id"] == (batch or world.cfg.anomalies.explain.holdout_batch) for f in fits)


def test_recompute_reports_a_one_ulp_difference_and_a_misplaced_gap(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = _partial(world, "dev")
    stored = world.points(t)
    h = min(stored)
    stored[h] = float(np.nextafter(np.float32(stored[h]), np.float32(np.inf)))
    gap = next(h for h in range(1, H + 1) if h not in stored)
    stored[gap] = 5000.0  # stored where the feature row is incomplete
    b = (
        _client(world, monkeypatch, stored)
        .get(f"{DEMAND}/recompute", params={"issue_time": _iso(t)})
        .json()
    )
    assert not b["matches_stored"] and b["leads_differing"] == 2
    assert 0 < b["max_abs_diff_mw"] < 0.01
    bad = {x["lead"] for x in b["leads"] if not x["matches"]}
    assert bad == {h, gap}
    assert b["leads"][gap - 1]["recomputed_mw"] is None


def test_a_stored_value_that_is_not_float32_is_refused() -> None:
    with pytest.raises(ValueError, match="float32"):
        rt.stored_array({1: 5000.1}, H)
    with pytest.raises(ValueError, match="outside"):
        rt.stored_array({0: 5000.0}, H)


# --------------------------------------------------------------------------- explain


@pytest.mark.parametrize("split", ["dev", "holdout"])
def test_explain_decomposes_every_stored_lead_with_its_feature_values(
    world: World, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    t = _partial(world, split)
    r = _client(world, monkeypatch).get(f"{DEMAND}/explain", params={"issue_time": _iso(t)})
    assert r.status_code == 200, r.text
    b = r.json()
    stored = world.stored[world.row(t)]
    have = np.flatnonzero(np.isfinite(stored)) + 1
    assert [x["lead"] for x in b["leads"]] == have.tolist()
    assert b["leads_without_forecast"] == [h for h in range(1, H + 1) if h not in set(have)]
    assert b["gate"]["passed"] and b["gate"]["n"] == len(have)
    builder = world.features.builder
    X = feature_rows(
        builder, "C", origin_positions(builder, pd.DatetimeIndex([t]).repeat(len(have))), have
    )
    forms = set()
    for i, x in enumerate(b["leads"]):
        feats = x["features"]
        total = x["anchor"]["mw"] + x["base_value_mw"] + sum(f["contribution_mw"] for f in feats)
        assert abs(total - x["forecast_float64_mw"]) < 1e-6
        assert np.float32(x["forecast_float64_mw"]) == np.float32(x["forecast_mw"])
        assert x["forecast_mw"] == float(stored[x["lead"] - 1])
        np.testing.assert_array_equal(np.array([f["value"] for f in feats], np.float32), X[i])
        by_family: dict[str, float] = {}
        for f in feats:
            by_family[f["family"]] = by_family.get(f["family"], 0.0) + f["contribution_mw"]
        for fam in x["families"]:
            assert abs(fam["contribution_mw"] - by_family[fam["family"]]) < 1e-9
        terms = x["anchor"]["terms"]
        assert terms == list(ANCHOR_TERMS[x["target_form"]])
        values = {f["name"]: f["value"] for f in feats}
        assert abs(x["anchor"]["mw"] - sum(values[n] for n in terms)) < 1e-9
        assert b["fits"][x["fit"]]["target_form"] == x["target_form"]
        forms.add(x["target_form"])
    assert sum(f["leads_recomputed"] for f in b["fits"]) == len(have)
    assert "not causes" in b["method"]


def test_the_fixture_mixes_target_forms(world: World) -> None:
    forms = {f.target_form for fits in world.features.fits.values() for f in fits}
    assert len(forms) >= 2, forms


def test_explain_one_lead_and_leads_without_a_forecast(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = _partial(world, "holdout")
    stored = world.points(t)
    client = _client(world, monkeypatch)
    have, gap = min(stored), next(h for h in range(1, H + 1) if h not in stored)
    one = client.get(f"{DEMAND}/explain", params={"issue_time": _iso(t), "lead": have}).json()
    assert [x["lead"] for x in one["leads"]] == [have] and one["leads_without_forecast"] == []
    assert len(one["fits"]) == 1
    r = client.get(f"{DEMAND}/explain", params={"issue_time": _iso(t), "lead": gap})
    assert r.status_code == 404 and f"lead {gap}" in r.json()["detail"]
    r = client.get(f"{DEMAND}/explain", params={"issue_time": _iso(t), "lead": H + 1})
    assert r.status_code == 422 and "at most" in r.json()["detail"]
    empty = _client(world, monkeypatch, stored={})
    r = empty.get(f"{DEMAND}/explain", params={"issue_time": _iso(t)})
    assert r.status_code == 404 and "no stored forecast" in r.json()["detail"]
    rc = empty.get(f"{DEMAND}/recompute", params={"issue_time": _iso(t)}).json()
    assert rc["leads_stored"] == 0 and not rc["matches_stored"]  # recomputed, not stored


def test_a_failed_gate_withholds_the_explanations_with_a_500(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = _partial(world, "dev")
    stored = world.points(t)
    h = max(stored)
    stored[h] = float(np.nextafter(np.float32(stored[h]), np.float32(-np.inf)))
    r = _client(world, monkeypatch, stored).get(f"{DEMAND}/explain", params={"issue_time": _iso(t)})
    assert r.status_code == 500
    body = r.json()
    assert set(body) == {"detail", "gate"} and "withheld" in body["detail"]
    assert body["gate"]["forecast_mismatches"] == 1 and not body["gate"]["passed"]
    # A stored forecast where the feature row is incomplete cannot be recomputed either.
    stored = world.points(t)
    stored[next(h for h in range(1, H + 1) if h not in stored)] = 5000.0
    r = _client(world, monkeypatch, stored).get(f"{DEMAND}/explain", params={"issue_time": _iso(t)})
    assert r.status_code == 500 and r.json()["gate"]["forecast_mismatches"] == 1


# --------------------------------------------------------------------------- loading


@pytest.fixture
def copied(world: World, tmp_path: Path) -> Iterator[World]:
    """The world with its model files copied to a data root the test may break."""
    root = tmp_path / "data"
    shutil.copytree(world.root / "models", root / "models")
    fits = {
        s: [dataclasses.replace(f, path=root / f.path.relative_to(world.root)) for f in v]
        for s, v in world.features.fits.items()
    }
    cfg = world.cfg.model_copy(deep=True)
    cfg.paths.data_root = root
    yield dataclasses.replace(
        world, root=root, cfg=cfg, features=rt.Features(world.features.builder, fits, 0.0)
    )


def test_a_missing_model_file_is_503_until_it_returns(
    copied: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = copied
    victim = w.features.fits["holdout"][0].path
    moved = victim.with_suffix(".bak")
    victim.rename(moved)
    client = _client(w, monkeypatch)
    t = _iso(w.split_times("dev")[0])
    r = client.get(f"{DEMAND}/recompute", params={"issue_time": t})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "model file not found: models/" in detail and str(w.root) not in detail
    runtime: rt.ModelRuntime = client.app.state.api.models  # type: ignore[attr-defined]
    checks = runtime.checks()
    assert checks["feature_context"][0] and not checks["model_files"][0]
    moved.rename(victim)
    assert client.get(f"{DEMAND}/recompute", params={"issue_time": t}).status_code == 503
    monkeypatch.setattr(rt, "RETRY_SECONDS", 0.0)  # the retry waits RETRY_SECONDS
    assert client.get(f"{DEMAND}/recompute", params={"issue_time": t}).status_code == 200
    assert runtime.checks()["model_files"][0]


def test_an_unmounted_data_root_is_named_without_its_path(
    copied: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.rmtree(copied.root)
    runtime = rt.ModelRuntime(copied.cfg, None, features=copied.features)
    with pytest.raises(rt.Unavailable, match="data root is not found"):
        runtime.require()


def test_the_features_need_a_database(world: World) -> None:
    runtime = rt.ModelRuntime(world.cfg, None)
    ok, detail = runtime.checks()["feature_context"]
    assert not ok and "no database engine" in detail
    assert not runtime.checks()["model_files"][0]


def test_boosters_are_cached_by_file_not_by_retrain_and_band(world: World, tmp_path: Path) -> None:
    """Both batches hold a ``2025-08-01`` fit per band: same key, different files."""
    fit = world.features.fits["dev"][0]
    twin_path = tmp_path / "other_batch" / fit.path.name
    twin_path.parent.mkdir()
    shutil.copy(fit.path, twin_path)
    twin = dataclasses.replace(fit, path=twin_path)
    assert twin.key == fit.key
    ours = rt.PathModelCache(world.cfg)
    assert ours.get(fit) is not ours.get(twin) and len(ours) == 2
    m4 = ModelCache(world.cfg)
    assert m4.get(fit) is m4.get(twin)  # why the API does not use M4's cache as is


def test_the_fits_must_be_the_published_batches(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = world.cfg.model_copy(deep=True)
    cfg.serving.sources["demand_point"]["dev"].batch_id = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(ValueError, match="dev: serving publishes"):
        rt.check_batches(cfg)
    rt.check_batches(world.cfg)
    client = _client(world, monkeypatch, dev_batch="11111111-1111-1111-1111-111111111111")
    t = _iso(world.split_times("dev")[0])
    r = client.get(f"{DEMAND}/explain", params={"issue_time": t})
    assert r.status_code == 503 and "whose fits are loaded" in r.json()["detail"]


def test_an_issue_time_without_a_fit_is_a_server_fault(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = dataclasses.replace(
        world, features=rt.Features(world.features.builder, {**world.features.fits, "dev": []}, 0)
    )
    r = _client(w, monkeypatch).get(
        f"{DEMAND}/recompute", params={"issue_time": _iso(w.split_times("dev")[0])}
    )
    assert r.status_code == 500 and "no stored dev fit" in r.json()["detail"]


def test_a_stored_value_that_is_not_float32_is_a_json_server_fault(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    t = _partial(world, "dev")
    stored = world.points(t)
    stored[min(stored)] += 1e-9  # not a float32 value: Postgres cannot hold this after P1
    for path in ("recompute", "explain"):
        r = _client(world, monkeypatch, stored).get(
            f"{DEMAND}/{path}", params={"issue_time": _iso(t)}
        )
        assert r.status_code == 500 and "not a float32 value" in r.json()["detail"]


# --------------------------------------------------------------------------- api-smoke


@pytest.mark.parametrize("split", ["dev", "holdout"])
def test_the_smoke_checks_pass_on_a_partly_stored_issue_time(
    world: World, monkeypatch: pytest.MonkeyPatch, split: str
) -> None:
    t = _partial(world, split).to_pydatetime()
    client = _client(world, monkeypatch)
    rc = smoke.check_recompute(client, t, repeats=2)
    assert rc.ok, rc.detail
    assert rc.target == f"{split} {_iso(pd.Timestamp(t))}" and len(rc.ms) == 2
    assert "0 differ" in rc.detail
    ex = smoke.check_explain(client, t, repeats=2)
    assert ex.ok, ex.detail
    assert "gate passed" in ex.detail and ex.kib is not None and ex.kib > 0


def test_the_smoke_checks_fail_on_a_one_ulp_difference(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    ts = _partial(world, "dev")
    stored = world.points(ts)
    h = max(stored)
    stored[h] = float(np.nextafter(np.float32(stored[h]), np.float32(-np.inf)))
    client = _client(world, monkeypatch, stored)
    rc = smoke.check_recompute(client, ts.to_pydatetime(), repeats=1)
    assert not rc.ok and "1 differ" in rc.detail
    ex = smoke.check_explain(client, ts.to_pydatetime(), repeats=1)
    assert not ex.ok and ex.detail.startswith("HTTP 500") and "'passed': False" in ex.detail
