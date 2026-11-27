"""``gridcast api-verify``: P2, P3, P5 and P6 against the running API (m5_plan.md §3; §4, step 5).

* **Refused** (nothing stored) unless ``/health`` is ``ok`` and its build is the image's, at the
  checkout's HEAD, with a clean image and a clean checkout (step 5's question round).
* **The truth is read without the API:** the parquet sources through publish's own reader
  (:func:`~gridcast.serving.sources.read_matrix`: interval bounds sorted by the frozen
  rearrangement), and ``publications``, ``market_intervals`` and ``anomalies`` by this
  module's own SQL.
* **Equal** means that a served number, cast to float32, has the source's bits and is itself
  exactly that float32; a null must fall exactly where the source has NaN. Each response's
  split, run ids, leads and target times are checked too. Every check runs to the end and
  keeps up to :data:`MAX_EXAMPLES` mismatches.
* **Samples** (one stream per draw from ``serving.verify.seed``): P2/P3 issue times per
  retrain fold of the published fits, among those with a stored lead; evolution targets on
  the 5-minute grid over (first issue, last issue + 24 h]; NEM days stored in full.
* **P5:** sequential client-side times (request sent → body read) on one keep-alive
  connection: the warm-ups, then every timed request in one seeded random order. p95 is
  numpy's default (linear) percentile; a timed response that is not 200 fails its endpoint.
* **P6** is judged here too: the served schema against ``docs/openapi.json``, a typed 200
  schema on every path, :data:`~gridcast.api.contract.INVALID_INPUTS` (422), a 404 table for
  the published data, and the suites' hypothesis search over query parameters (no 500).

The result, every latency included, goes to ``<data root>/serving/verify_<UTC time>.json`` and
is never overwritten; ``gridcast report-m5`` renders the stored runs.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.api.contract import INVALID_INPUTS, SEARCH_EXAMPLES, SEARCH_LITERALS
from gridcast.api.contract import SEARCH_PARAMETERS as SEARCH_PARAMS
from gridcast.config import SERVING_SPLITS, AppConfig, ServingVerifyConfig
from gridcast.forecasting.origins import NEM_TZ, nem_midnight_utc
from gridcast.logs import get_logger
from gridcast.serving.sources import INTERVAL_COLUMNS, PRODUCTS, SourceMatrix

log = get_logger(__name__)

MAX_EXAMPLES = 20
DEMAND = "/v1/forecasts/demand"
EVOLUTION = "/v1/forecasts/demand/evolution"
RECOMPUTE = "/v1/forecasts/demand/recompute"
EXPLAIN = "/v1/forecasts/demand/explain"
PRICE = "/v1/forecasts/price"
ACTUALS = "/v1/actuals"
ANOMALIES = "/v1/anomalies"
EVENT = "/v1/anomalies/{}"
CONTAINERS = ("gridcast-api", "gridcast-db")
# Wider than every stored event: /v1/anomalies pages through all of them inside it.
EVERY_EVENT = {"start": "2000-01-01T00:00:00Z", "end": "2100-01-01T00:00:00Z"}
ANOMALY_LIMIT = 100  # the endpoint's default page size (P2) and P5's month request
STORED_ENDPOINTS = ("forecast", "evolution", "price", "actuals", "anomalies", "event")
MODEL_ENDPOINTS = ("recompute", "explain")
SUMMARY_FIELDS = (
    "event_id",
    "run_id",
    "split",
    "first_interval_end",
    "last_interval_end",
    "peak_interval_end",
    "tier",
    "detectors",
    "categories",
    "min_p",
    "lines_only",
    "summary",
)
ACTUAL_FIELDS = {  # served field -> market_intervals column
    "interval_start": "interval_start",
    "interval_end": "interval_end",
    "demand_mw": "total_demand_mw",
    "price_aud_mwh": "rrp_aud_mwh",
    "price_status": "price_status",
    "apc_flag_bits": "apc_flag_bits",
    "administered_price_binding": "administered_price_binding",
    "market_cap_floor_binding": "market_cap_floor_binding",
    "price_manually_overwritten": "price_manually_overwritten",
    "market_suspended": "market_suspended",
}


class VerifyRefused(Exception):
    """A precondition failed; nothing was measured or stored."""


def iso(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(value: Any) -> pd.Timestamp | None:
    """A served timestamp, or None if it is missing or not one."""
    if not isinstance(value, str):
        return None
    try:
        t = pd.Timestamp(value)
    except ValueError:
        return None
    return None if t.tzinfo is None else t.tz_convert("UTC")


def _stream(seed: int, name: str) -> np.random.Generator:
    """An independent generator for one draw (so adding a draw changes no other)."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


# --------------------------------------------------------------------------- comparisons


def float32_mismatches(served: Sequence[Any], expected: np.ndarray) -> np.ndarray:
    """Per value, whether ``served`` differs from the stored float32 ``expected``.

    Equal means: null where ``expected`` is NaN; otherwise a JSON number that is exactly a
    float32 value with ``expected``'s bits.
    """
    want = np.asarray(expected, dtype=np.float32)
    if len(served) != len(want):
        raise ValueError(f"{len(served)} served values for {len(want)} expected")
    null = np.array([v is None for v in served], dtype=bool)
    number = np.array(
        [isinstance(v, int | float) and not isinstance(v, bool) for v in served], dtype=bool
    )
    values = np.array(
        [float(v) if ok else np.nan for v, ok in zip(served, number, strict=True)],
        dtype=np.float64,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        as32 = values.astype(np.float32)
    exact = as32.astype(np.float64) == values  # False for NaN, and for an overflow to inf
    want_nan = np.isnan(want)
    same_bits = as32.view(np.uint32) == want.view(np.uint32)
    ok = np.where(null, want_nan, number & ~want_nan & exact & same_bits)
    return ~ok


def _equal(served: Any, expected: Any) -> bool:
    """A served JSON value against a database value, exactly."""
    if expected is None:
        return served is None
    if isinstance(expected, dt.datetime | pd.Timestamp):
        t = _ts(served)
        return t is not None and t == pd.Timestamp(expected).tz_convert("UTC")
    if isinstance(expected, bool):
        return isinstance(served, bool) and served == expected
    if isinstance(expected, float):
        ok = isinstance(served, int | float) and not isinstance(served, bool)
        return ok and float(served) == expected
    if isinstance(expected, int):
        return isinstance(served, int) and not isinstance(served, bool) and served == expected
    if isinstance(expected, list | tuple):
        return isinstance(served, list) and served == list(expected)
    return bool(served == expected)


@dataclass(slots=True)
class Tally:
    """One check: requests made, values and fields compared, what differed."""

    criterion: str
    name: str
    requests: int = 0
    values_compared: int = 0  # served numbers compared with a stored float32
    fields_compared: int = 0  # every other comparison
    mismatches: int = 0
    bad_responses: int = 0  # an unexpected status, a transport error, or not JSON
    examples: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def note(self, text_: str) -> None:
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(text_)

    def check(self, ok: bool, what: str) -> bool:
        self.fields_compared += 1
        if not ok:
            self.mismatches += 1
            self.note(what)
        return ok

    def numbers(
        self, served: Sequence[Any], expected: np.ndarray, ctx: str, keys: Sequence[str]
    ) -> np.ndarray:
        """Compares ``served`` with ``expected`` (float32), value ``i`` named ``keys[i]``;
        returns the per-value mismatches."""
        if not len(served) == len(expected) == len(keys):
            self.check(False, f"{ctx}: {len(served)} values served, {len(expected)} stored")
            return np.ones(len(expected), dtype=bool)
        bad = float32_mismatches(served, expected)
        self.values_compared += len(bad)
        self.mismatches += int(bad.sum())
        for i in np.flatnonzero(bad)[: max(0, MAX_EXAMPLES - len(self.examples))]:
            self.note(f"{ctx} {keys[i]}: served {served[i]!r}, stored {float(expected[i])!r}")
        return bad

    @property
    def passed(self) -> bool:
        return self.requests > 0 and self.mismatches == 0 and self.bad_responses == 0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def fetch(
    client: httpx.Client,
    tally: Tally,
    path: str,
    params: dict[str, str] | None = None,
    expect: int = 200,
) -> Any | None:
    """GET ``path``; the JSON body if the status is ``expect``, else None (counted)."""
    tally.requests += 1
    where = f"{path} {params or {}}"
    try:
        r = client.get(path, params=params)
    except httpx.HTTPError as exc:
        tally.bad_responses += 1
        tally.note(f"{where}: {type(exc).__name__}: {exc}")
        return None
    if r.status_code != expect:
        tally.bad_responses += 1
        tally.note(f"{where}: HTTP {r.status_code}, expected {expect}: {r.text[:200]}")
        return None
    try:
        return r.json()
    except ValueError:
        tally.bad_responses += 1
        tally.note(f"{where}: the body is not JSON")
        return None


def _error_body(body: Any) -> bool:
    return isinstance(body, dict) and set(body) == {"detail"} and isinstance(body["detail"], str)


# --------------------------------------------------------------------------- the truth


@dataclass(frozen=True, slots=True)
class Pub:
    """A publication's run and issue-time range (its latest attempt, which succeeded)."""

    run_id: int
    first: pd.Timestamp
    last: pd.Timestamp

    def covers(self, t: pd.Timestamp) -> bool:
        return bool(self.first <= t <= self.last)


@dataclass(frozen=True, slots=True)
class Fold:
    """One retrain fold of the published fits: issue times [start, end) of a split."""

    split: str
    retrain: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(slots=True)
class Truth:
    """What the API should serve, read without it."""

    region: str
    n_leads: int
    step: int  # minutes between leads
    issue_every: int  # minutes between issue times
    holdout_start: pd.Timestamp
    pubs: dict[tuple[str, str], Pub]
    matrices: dict[tuple[str, str], SourceMatrix]
    folds: list[Fold] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)  # served events, in order
    actual_days: list[pd.Timestamp] = field(default_factory=list)  # NEM midnights (UTC)
    actuals_first_end: pd.Timestamp | None = None
    other_event: int | None = None  # an event of a run that is not served (v1)
    actuals: Callable[[pd.Timestamp, pd.Timestamp], list[dict[str, Any]]] | None = None
    _index: dict[tuple[str, str], pd.Index] = field(default_factory=dict)

    def split_of(self, t: pd.Timestamp) -> str:
        return "dev" if t < self.holdout_start else "holdout"

    def rows(
        self, product: str, split: str, times: Sequence[pd.Timestamp]
    ) -> dict[str, np.ndarray]:
        """Each column's (len(times), H) float32 values; NaN rows where nothing is stored."""
        columns = PRODUCTS[product].columns
        out = {c: np.full((len(times), self.n_leads), np.nan, dtype=np.float32) for c in columns}
        m = self.matrices.get((product, split))
        if m is None or not len(times):
            return out
        key = (product, split)
        if key not in self._index:
            self._index[key] = pd.Index(m.times.as_unit("us").asi8)
        utc = [pd.Timestamp(t).tz_convert("UTC") for t in times]
        want = pd.Index(pd.DatetimeIndex(utc).as_unit("us").asi8)
        pos = self._index[key].get_indexer(want)
        hit = pos >= 0
        for c in columns:
            out[c][hit] = m.by_column[c][pos[hit]]
        return out

    def point(self, t: pd.Timestamp) -> np.ndarray:
        return self.rows("demand_point", self.split_of(t), [t])["point"][0]

    def stored_times(self, split: str) -> pd.DatetimeIndex:
        """Published demand issue times of ``split`` with at least one stored lead."""
        m = self.matrices[("demand_point", split)]
        return m.times[np.isfinite(m.by_column["point"]).any(axis=1)]

    def published(self, product: str) -> pd.DatetimeIndex:
        """Every published issue time of a product, both splits (its ranges on the grid)."""
        parts = [
            pd.date_range(p.first, p.last, freq=f"{self.issue_every}min")
            for s in SERVING_SPLITS
            if (p := self.pubs.get((product, s))) is not None
        ]
        return (
            pd.DatetimeIndex(parts[0].append(parts[1:]))
            if parts
            else pd.DatetimeIndex([], tz="UTC")
        )

    def demand_span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        pubs = [self.pubs[("demand_point", s)] for s in SERVING_SPLITS]
        return min(p.first for p in pubs), max(p.last for p in pubs)

    def evolution_issues(self, target: pd.Timestamp) -> list[pd.Timestamp]:
        """The published issue times in [target − H steps, target)."""
        lo = target - pd.Timedelta(minutes=self.n_leads * self.step)
        grid = pd.date_range(
            lo.ceil(f"{self.issue_every}min"), target, freq=f"{self.issue_every}min"
        )
        out = []
        for t in grid[grid < target]:
            p = self.pubs.get(("demand_point", self.split_of(t)))
            if p is not None and p.covers(t):
                out.append(t)
        return out


def published_runs(engine: Engine, region: str) -> dict[tuple[str, str], Pub]:
    """The latest publication of each (product, split); refuses unless each succeeded."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT ON (product, split) product, split, run_id, first_issue,
                       last_issue, status
                FROM publications WHERE region_id = :r
                ORDER BY product, split, publication_id DESC
                """
            ),
            {"r": region},
        ).all()
    out, bad = {}, []
    for r in rows:
        if r.status != "success":
            bad.append(f"{r.product}/{r.split}: {r.status}")
        out[(r.product, r.split)] = Pub(
            int(r.run_id), pd.Timestamp(r.first_issue), pd.Timestamp(r.last_issue)
        )
    missing = [f"{p}/{s}" for p in PRODUCTS for s in SERVING_SPLITS if (p, s) not in out]
    if bad or missing:
        raise VerifyRefused(
            "publications incomplete: " + "; ".join(bad + [f"{m}: never" for m in missing])
        )
    return out


def anomaly_rows(engine: Engine, run_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Every event of the served runs, with its evidence, in the API's documented order."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT {', '.join(SUMMARY_FIELDS)}, evidence FROM anomalies "
                "WHERE run_id = ANY(:runs) ORDER BY first_interval_end, event_id"
            ),
            {"runs": list(run_ids)},
        ).all()
    return [dict(r._mapping) for r in rows]


def other_event(engine: Engine, run_ids: Sequence[int]) -> int | None:
    with engine.connect() as conn:
        v = conn.execute(
            text("SELECT min(event_id) FROM anomalies WHERE NOT (run_id = ANY(:runs))"),
            {"runs": list(run_ids)},
        ).scalar_one()
    return None if v is None else int(v)


def actual_rows(
    engine: Engine, region: str, start: pd.Timestamp, end: pd.Timestamp
) -> list[dict[str, Any]]:
    """``market_intervals`` rows ending in (start, end], in order (this module's own SQL)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT {', '.join(ACTUAL_FIELDS.values())} FROM market_intervals "
                "WHERE region_id = :r AND interval_end > :lo AND interval_end <= :hi "
                "ORDER BY interval_end"
            ),
            {"r": region, "lo": start.to_pydatetime(), "hi": end.to_pydatetime()},
        ).all()
    return [dict(r._mapping) for r in rows]


def full_days(engine: Engine, region: str, step: int) -> tuple[list[pd.Timestamp], Any]:
    """NEM days with every interval stored (their UTC starts), and the first interval end."""
    with engine.connect() as conn:
        ends = [
            r[0]
            for r in conn.execute(
                text("SELECT interval_end FROM market_intervals WHERE region_id = :r"),
                {"r": region},
            ).all()
        ]
    if not ends:
        return [], None
    idx = pd.DatetimeIndex(ends).tz_convert("UTC")
    nem_day = (idx - pd.Timedelta(minutes=step)).tz_convert(NEM_TZ).tz_localize(None).normalize()
    counts = pd.Series(1, index=nem_day).groupby(level=0).size()
    per_day = 24 * 60 // step
    days = [nem_midnight_utc(d) for d in counts.index[counts.to_numpy() == per_day]]
    return days, idx.min()


def fits_folds(engine: Engine, cfg: AppConfig, pubs: dict[tuple[str, str], Pub]) -> list[Fold]:
    """The retrain folds of the fits the API loads, clipped to each split's published range."""
    from gridcast.api.runtime import split_fits

    step = pd.Timedelta(minutes=cfg.forecast.issue_every_minutes)
    out = []
    for split in SERVING_SPLITS:
        pub = pubs[("demand_point", split)]
        spans: dict[str, tuple[pd.Timestamp, pd.Timestamp | None]] = {}
        for f in split_fits(engine, cfg, split):
            span = (f.fold_start, f.fold_end)
            if spans.setdefault(f.retrain, span) != span:
                raise ValueError(f"{split} retrain {f.retrain}: its bands' folds differ")
        for retrain, (a, b) in sorted(spans.items()):
            start = max(a, pub.first)
            end = pub.last + step if b is None else min(b, pub.last + step)
            if start < end:
                out.append(Fold(split, retrain, start, end))
    return out


def load_truth(engine: Engine, cfg: AppConfig, folds: list[Fold] | None = None) -> Truth:
    """Every source matrix (publish's reader), the publications, events and actual days."""
    from gridcast.serving.sources import read_matrix, resolve_sources, windows

    region = cfg.serving.region
    n_leads, issue_every = cfg.forecast.horizon_steps, cfg.forecast.issue_every_minutes
    step = cfg.nem.dispatch_interval_minutes
    pubs = published_runs(engine, region)
    matrices = {
        (s.product.name, s.split): read_matrix(s, n_leads, issue_every)
        for s in resolve_sources(engine, cfg)
    }
    runs = [r.run_id for r in cfg.serving.anomaly_runs.values()]
    days, first_end = full_days(engine, region, step)
    return Truth(
        region=region,
        n_leads=n_leads,
        step=step,
        issue_every=issue_every,
        holdout_start=windows(cfg)["holdout"].start,
        pubs=pubs,
        matrices=matrices,
        folds=folds if folds is not None else fits_folds(engine, cfg, pubs),
        anomalies=anomaly_rows(engine, runs),
        actual_days=days,
        actuals_first_end=None if first_end is None else pd.Timestamp(first_end),
        other_event=other_event(engine, runs),
        actuals=lambda lo, hi: actual_rows(engine, region, lo, hi),
    )


# --------------------------------------------------------------------------- samples


@dataclass(slots=True)
class Samples:
    by_fold: list[tuple[Fold, list[pd.Timestamp]]]
    candidates: list[int]  # per fold: its issue times with a stored lead
    targets: list[pd.Timestamp]
    days: list[pd.Timestamp]

    @property
    def issue_times(self) -> list[pd.Timestamp]:
        return [t for _, ts in self.by_fold for t in ts]

    def as_dict(self) -> dict[str, Any]:
        return {
            "folds": [
                {
                    "split": f.split,
                    "retrain": f.retrain,
                    "start": iso(f.start),
                    "end": iso(f.end),
                    "candidates": n,
                    "issue_times": [iso(t) for t in ts],
                }
                for (f, ts), n in zip(self.by_fold, self.candidates, strict=True)
            ],
            "evolution_targets": [iso(t) for t in self.targets],
            "actual_days": [iso(t) for t in self.days],
        }


def _choose(rng: np.random.Generator, items: Sequence[Any], k: int) -> list[Any]:
    """``k`` of ``items`` without replacement (all if fewer), in their order."""
    if len(items) <= k:
        return list(items)
    pick = np.sort(rng.choice(len(items), size=k, replace=False))
    return [items[int(i)] for i in pick]


def evolution_domain(truth: Truth) -> tuple[pd.Timestamp, int]:
    """Targets first + k·step for k = 1…n: (first issue, last issue + H steps]."""
    first, last = truth.demand_span()
    n = int((last - first) / pd.Timedelta(minutes=truth.step)) + truth.n_leads
    return first, n


def draw_samples(truth: Truth, vcfg: ServingVerifyConfig) -> Samples:
    rng = _stream(vcfg.seed, "issue_times")
    by_fold, candidates = [], []
    for fold in truth.folds:
        stored = truth.stored_times(fold.split)
        inside = stored[(stored >= fold.start) & (stored < fold.end)]
        candidates.append(len(inside))
        by_fold.append((fold, _choose(rng, list(inside), vcfg.issue_times_per_fold)))
    first, n = evolution_domain(truth)
    ks = _stream(vcfg.seed, "evolution").choice(
        n, size=min(n, vcfg.evolution_targets), replace=False
    )
    targets = [first + pd.Timedelta(minutes=truth.step * (int(k) + 1)) for k in np.sort(ks)]
    days = _choose(_stream(vcfg.seed, "actual_days"), truth.actual_days, vcfg.actual_days)
    return Samples(by_fold, candidates, targets, days)


# --------------------------------------------------------------------------- P2


def _lead_keys(n_leads: int) -> list[str]:
    return [f"lead {h}" for h in range(1, n_leads + 1)]


def _lead_fields(
    tally: Tally, ctx: str, leads: Any, t: pd.Timestamp, n_leads: int, step: int
) -> bool:
    if not tally.check(
        isinstance(leads, list) and len(leads) == n_leads,
        f"{ctx}: {len(leads) if isinstance(leads, list) else 'no'} leads, expected {n_leads}",
    ):
        return False
    hs = list(range(1, n_leads + 1))
    tally.check([x.get("lead") for x in leads] == hs, f"{ctx}: leads not 1…{n_leads} in order")
    tally.check(
        [x.get("lead_minutes") for x in leads] == [h * step for h in hs],
        f"{ctx}: lead_minutes differ",
    )
    tally.check(
        [_ts(x.get("target_end")) for x in leads]
        == [t + pd.Timedelta(minutes=h * step) for h in hs],
        f"{ctx}: target_end differs",
    )
    return True


def check_demand(client: httpx.Client, truth: Truth, times: Sequence[pd.Timestamp]) -> Tally:
    tally = Tally("P2", "demand forecasts")
    for t in times:
        split = truth.split_of(t)
        ctx = f"demand {iso(t)}"
        body = fetch(client, tally, DEMAND, {"issue_time": iso(t)})
        if body is None:
            continue
        tally.check(body.get("split") == split, f"{ctx}: split {body.get('split')!r}")
        tally.check(_ts(body.get("issue_time")) == t, f"{ctx}: issue_time differs")
        for key, product in (
            ("point_run_id", "demand_point"),
            ("interval_run_id", "demand_interval"),
        ):
            want = truth.pubs[(product, split)].run_id
            tally.check(body.get(key) == want, f"{ctx}: {key} {body.get(key)!r}, expected {want}")
        leads = body.get("leads")
        if not _lead_fields(tally, ctx, leads, t, truth.n_leads, truth.step):
            continue
        point = truth.rows("demand_point", split, [t])["point"][0]
        keys = _lead_keys(truth.n_leads)
        tally.numbers([x.get("point_mw") for x in leads], point, ctx, keys)
        bounds = truth.rows("demand_interval", split, [t])
        for col in INTERVAL_COLUMNS:
            tally.numbers([x.get(f"{col}_mw") for x in leads], bounds[col][0], f"{ctx} {col}", keys)
    return tally


def check_price(client: httpx.Client, truth: Truth, times: Sequence[pd.Timestamp]) -> Tally:
    """Both baselines at each issue time; 404 where the price baselines are not published."""
    tally = Tally("P2", "price baselines")
    unpublished = 0
    names = (("persistence", "price_persistence"), ("median_profile", "price_median_profile"))
    for t in times:
        split = truth.split_of(t)
        ctx = f"price {iso(t)}"
        pubs = {name: truth.pubs[(product, split)] for name, product in names}
        if not all(p.covers(t) for p in pubs.values()):
            unpublished += 1
            body = fetch(client, tally, PRICE, {"issue_time": iso(t)}, expect=404)
            if body is not None:
                tally.check(_error_body(body), f"{ctx}: 404 body {body!r}")
            continue
        body = fetch(client, tally, PRICE, {"issue_time": iso(t)})
        if body is None:
            continue
        tally.check(body.get("split") == split, f"{ctx}: split {body.get('split')!r}")
        tally.check(_ts(body.get("issue_time")) == t, f"{ctx}: issue_time differs")
        for name, _ in names:
            key = f"{name}_run_id"
            want = pubs[name].run_id
            tally.check(body.get(key) == want, f"{ctx}: {key} {body.get(key)!r}, expected {want}")
        leads = body.get("leads")
        if not _lead_fields(tally, ctx, leads, t, truth.n_leads, truth.step):
            continue
        for name, product in names:
            tally.numbers(
                [x.get(f"{name}_aud_mwh") for x in leads],
                truth.rows(product, split, [t])["point"][0],
                f"{ctx} {name}",
                _lead_keys(truth.n_leads),
            )
    tally.extra["not_published"] = unpublished
    return tally


def check_evolution(client: httpx.Client, truth: Truth, targets: Sequence[pd.Timestamp]) -> Tally:
    """Every published issue time of [T − 24 h, T), each with its lead, point and bounds."""
    tally = Tally("P2", "forecast evolution")
    items_seen = 0
    step = pd.Timedelta(minutes=truth.step)
    for target in targets:
        ctx = f"evolution {iso(target)}"
        issues = truth.evolution_issues(target)
        if not issues:
            body = fetch(client, tally, EVOLUTION, {"target_end": iso(target)}, expect=404)
            if body is not None:
                tally.check(_error_body(body), f"{ctx}: 404 body {body!r}")
            continue
        body = fetch(client, tally, EVOLUTION, {"target_end": iso(target)})
        if body is None:
            continue
        tally.check(_ts(body.get("target_end")) == target, f"{ctx}: target_end differs")
        items = body.get("items")
        if not tally.check(
            isinstance(items, list) and [_ts(x.get("issue_time")) for x in items] == issues,
            f"{ctx}: served issue times differ from the {len(issues)} published",
        ):
            continue
        items_seen += len(items)
        for split in SERVING_SPLITS:
            pos = [k for k, t in enumerate(issues) if truth.split_of(t) == split]
            if not pos:
                continue
            ts = [issues[k] for k in pos]
            got = [items[k] for k in pos]
            leads = [int((target - t) / step) for t in ts]
            runs = {
                "point_run_id": truth.pubs[("demand_point", split)].run_id,
                "interval_run_id": truth.pubs[("demand_interval", split)].run_id,
            }
            tally.check(all(x.get("split") == split for x in got), f"{ctx}: split differs")
            for key, want in runs.items():
                tally.check(all(x.get(key) == want for x in got), f"{ctx}: {key} differs")
            tally.check([x.get("lead") for x in got] == leads, f"{ctx}: lead differs")
            tally.check(
                [x.get("lead_minutes") for x in got] == [h * truth.step for h in leads],
                f"{ctx}: lead_minutes differs",
            )
            col = np.asarray(leads) - 1
            rows = np.arange(len(ts))
            point = truth.rows("demand_point", split, ts)["point"][rows, col]
            keys = [f"issued {iso(t)}" for t in ts]
            tally.numbers([x.get("point_mw") for x in got], point, ctx, keys)
            bounds = truth.rows("demand_interval", split, ts)
            for c in INTERVAL_COLUMNS:
                tally.numbers(
                    [x.get(f"{c}_mw") for x in got], bounds[c][rows, col], f"{ctx} {c}", keys
                )
    tally.extra["items"] = items_seen
    return tally


def check_actuals(client: httpx.Client, truth: Truth, days: Sequence[pd.Timestamp]) -> Tally:
    """Each NEM day's 5-minute intervals, field by field, against ``market_intervals``."""
    tally = Tally("P2", "actuals")
    if truth.actuals is None:
        raise ValueError("the truth has no actuals reader")
    per_day = 24 * 60 // truth.step
    intervals = 0
    for lo in days:
        hi = lo + pd.Timedelta(days=1)
        ctx = f"actuals {iso(lo)}"
        want = truth.actuals(lo, hi)
        body = fetch(client, tally, ACTUALS, {"start": iso(lo), "end": iso(hi)})
        if body is None:
            continue
        got = body.get("intervals")
        if not tally.check(
            isinstance(got, list) and len(got) == len(want) == per_day,
            f"{ctx}: {len(got) if isinstance(got, list) else 'no'} intervals served, "
            f"{len(want)} stored, {per_day} expected",
        ):
            continue
        intervals += len(got)
        for s, w in zip(got, want, strict=True):
            for served, column in ACTUAL_FIELDS.items():
                tally.check(
                    _equal(s.get(served), w[column]),
                    f"{ctx} {iso(w['interval_end'])} {served}: served {s.get(served)!r}, "
                    f"stored {w[column]!r}",
                )
    tally.extra["intervals"] = intervals
    return tally


def _summary_fields(tally: Tally, ctx: str, served: Any, row: dict[str, Any]) -> None:
    if not tally.check(isinstance(served, dict), f"{ctx}: not an object"):
        return
    for name in SUMMARY_FIELDS:
        tally.check(
            _equal(served.get(name), row[name]),
            f"{ctx} {name}: served {served.get(name)!r}, stored {row[name]!r}",
        )


def check_anomalies(
    client: httpx.Client, truth: Truth, limit: int = ANOMALY_LIMIT, by_id: bool = True
) -> Tally:
    """Every served event paged through (keyset cursor), then each event by id with evidence."""
    tally = Tally("P2", "anomaly events")
    served: list[Any] = []
    after: str | None = None
    pages = 0
    max_pages = len(truth.anomalies) // limit + 2
    while True:
        params = {**EVERY_EVENT, "limit": str(limit)}
        if after is not None:
            params["after"] = after
        body = fetch(client, tally, ANOMALIES, params)
        if body is None:
            break
        pages += 1
        events = body.get("events") or []
        tally.check(len(events) <= limit, f"page {pages}: {len(events)} events > limit {limit}")
        served += events
        after = body.get("next")
        if after is None:
            break
        if not tally.check(pages < max_pages, f"paging did not end after {pages} pages"):
            break
    tally.check(
        len(served) == len(truth.anomalies),
        f"{len(served)} events paged, {len(truth.anomalies)} stored",
    )
    for k, (s, row) in enumerate(zip(served, truth.anomalies, strict=False)):
        _summary_fields(tally, f"page item {k} (event {row['event_id']})", s, row)
    if by_id:
        for row in truth.anomalies:
            ctx = f"event {row['event_id']}"
            body = fetch(client, tally, EVENT.format(row["event_id"]), None)
            if body is None:
                continue
            _summary_fields(tally, ctx, body, row)
            tally.check(body.get("evidence") == row["evidence"], f"{ctx}: evidence differs")
    tally.extra.update(pages=pages, events=len(truth.anomalies), limit=limit)
    return tally


# --------------------------------------------------------------------------- P3


def check_recompute(client: httpx.Client, truth: Truth, times: Sequence[pd.Timestamp]) -> Tally:
    """Each lead recomputed equals the stored float32 (null where nothing was stored)."""
    tally = Tally("P3", "recompute")
    pairs = equal = 0
    for t in times:
        split = truth.split_of(t)
        ctx = f"recompute {iso(t)}"
        want = truth.point(t)
        pairs += len(want)
        body = fetch(client, tally, RECOMPUTE, {"issue_time": iso(t)})
        if body is None:
            continue
        n_stored = int(np.isfinite(want).sum())
        tally.check(body.get("split") == split, f"{ctx}: split {body.get('split')!r}")
        tally.check(_ts(body.get("issue_time")) == t, f"{ctx}: issue_time differs")
        run = truth.pubs[("demand_point", split)].run_id
        tally.check(body.get("point_run_id") == run, f"{ctx}: point_run_id differs")
        tally.check(body.get("matches_stored") is True, f"{ctx}: matches_stored is not true")
        tally.check(
            body.get("leads_differing") == 0, f"{ctx}: {body.get('leads_differing')} differ"
        )
        for key in ("leads_stored", "leads_recomputed"):
            tally.check(body.get(key) == n_stored, f"{ctx}: {key} {body.get(key)}, {n_stored}")
        leads = body.get("leads")
        if not _lead_fields(tally, ctx, leads, t, truth.n_leads, truth.step):
            continue
        keys = _lead_keys(truth.n_leads)
        bad = tally.numbers(
            [x.get("recomputed_mw") for x in leads], want, f"{ctx} recomputed", keys
        )
        bad |= tally.numbers([x.get("stored_mw") for x in leads], want, f"{ctx} stored", keys)
        flags = np.array([x.get("matches") is True for x in leads], dtype=bool)
        tally.fields_compared += len(flags)
        tally.mismatches += int((~flags).sum())
        equal += int((~bad & flags).sum())
    tally.extra.update(pairs=pairs, pairs_equal=equal)
    return tally


def check_explain(
    client: httpx.Client, truth: Truth, times: Sequence[pd.Timestamp], tolerance: float
) -> Tally:
    """Every stored lead explained with E1's gate passed; each lead's decomposition re-added."""
    tally = Tally("P3", "explain")
    explained_times = explained_leads = 0
    for t in times:
        split = truth.split_of(t)
        ctx = f"explain {iso(t)}"
        want = truth.point(t)
        stored = [h for h in range(1, len(want) + 1) if np.isfinite(want[h - 1])]
        before = tally.mismatches
        body = fetch(client, tally, EXPLAIN, {"issue_time": iso(t)})
        if body is None:
            continue
        gate = body.get("gate") or {}
        tally.check(body.get("split") == split, f"{ctx}: split {body.get('split')!r}")
        tally.check(_ts(body.get("issue_time")) == t, f"{ctx}: issue_time differs")
        run = truth.pubs[("demand_point", split)].run_id
        tally.check(body.get("point_run_id") == run, f"{ctx}: point_run_id differs")
        tally.check(gate.get("passed") is True, f"{ctx}: gate {gate}")
        tally.check(gate.get("n") == len(stored), f"{ctx}: gate n {gate.get('n')}, {len(stored)}")
        for key in ("forecast_mismatches", "sum_failures"):
            tally.check(gate.get(key) == 0, f"{ctx}: gate {key} {gate.get(key)}")
        have = set(stored)
        missing = [h for h in range(1, len(want) + 1) if h not in have]
        tally.check(body.get("leads_without_forecast") == missing, f"{ctx}: leads_without_forecast")
        leads = body.get("leads")
        if not tally.check(
            isinstance(leads, list) and [x.get("lead") for x in leads] == stored,
            f"{ctx}: explained leads differ from the {len(stored)} stored",
        ):
            continue
        idx = np.asarray(stored) - 1
        keys = [f"lead {h}" for h in stored]
        tally.numbers([x.get("forecast_mw") for x in leads], want[idx], ctx, keys)
        for k, x in enumerate(leads):
            where = f"{ctx} lead {stored[k]}"
            try:
                f64 = float(x["forecast_float64_mw"])
                features = [float(c["contribution_mw"]) for c in x["features"]]
                families = [float(c["contribution_mw"]) for c in x["families"]]
                total = float(x["anchor"]["mw"]) + float(x["base_value_mw"]) + math.fsum(features)
            except (KeyError, TypeError, ValueError) as exc:
                tally.check(False, f"{where}: malformed ({type(exc).__name__}: {exc})")
                continue
            tally.check(
                np.float32(f64).view(np.uint32) == want[idx[k]].view(np.uint32),
                f"{where}: forecast_float64_mw {f64!r} is not the stored float32",
            )
            tally.check(
                abs(total - f64) <= tolerance,
                f"{where}: anchor + base + Σ contributions − forecast = {total - f64:.3g}",
            )
            tally.check(
                abs(math.fsum(families) - math.fsum(features)) <= tolerance,
                f"{where}: families sum to {math.fsum(families)!r}, features to "
                f"{math.fsum(features)!r}",
            )
        explained_leads += len(leads)
        explained_times += int(tally.mismatches == before)
    tally.extra.update(issue_times=len(times), explained=explained_times, leads=explained_leads)
    return tally


# --------------------------------------------------------------------------- P6


def check_schema(client: httpx.Client, committed: dict[str, Any]) -> Tally:
    """The served schema equals the committed one; every path is GET with a typed 200."""
    tally = Tally("P6", "schema")
    body = fetch(client, tally, "/openapi.json")
    if body is None:
        return tally
    differs = sorted(k for k in set(body) | set(committed) if body.get(k) != committed.get(k))
    tally.check(not differs, f"the served schema differs from docs/openapi.json in {differs}")
    paths = body.get("paths") or {}
    for path, ops in sorted(paths.items()):
        tally.check(set(ops) == {"get"}, f"{path}: methods {sorted(ops)}")
        ok = (ops.get("get") or {}).get("responses", {}).get("200", {})
        schema = ok.get("content", {}).get("application/json", {}).get("schema", {})
        tally.check("$ref" in schema, f"{path}: its 200 response has no typed schema")
    tally.extra["paths"] = len(paths)
    return tally


def check_errors(
    client: httpx.Client,
    name: str,
    table: Sequence[tuple[str, dict[str, str], str]],
    status: int,
) -> Tally:
    """Each request answers ``status`` with ``{"detail": <message>}`` (containing the text for
    422; for 404 the third item only names the case)."""
    tally = Tally("P6", name)
    for path, params, text_ in table:
        body = fetch(client, tally, path, params, expect=status)
        if body is None:
            continue
        ok = _error_body(body) and (status != 422 or text_ in body["detail"])
        tally.check(
            ok, f"{path} {params}: body {body!r}" + (f", wants {text_!r}" if status == 422 else "")
        )
    tally.extra["cases"] = len(table)
    return tally


def not_published_table(truth: Truth) -> list[tuple[str, dict[str, str], str]]:
    """Valid requests with nothing published there (404), from the published data."""
    half = pd.Timedelta(minutes=truth.issue_every)
    first, last = truth.demand_span()
    price_last = max(truth.pubs[("price_persistence", s)].last for s in SERVING_SPLITS)
    rows = [
        (DEMAND, {"issue_time": iso(first - half)}, "demand before the published range"),
        (DEMAND, {"issue_time": iso(last + half)}, "demand after the published range"),
        (DEMAND, {"issue_time": iso(first), "region": "NSW1"}, "an unpublished region"),
        (EVOLUTION, {"target_end": iso(first)}, "a target before every issue time"),
        (PRICE, {"issue_time": iso(price_last + half)}, "price after its last issue time"),
        (RECOMPUTE, {"issue_time": iso(first - half)}, "recompute before the published range"),
        (EXPLAIN, {"issue_time": iso(first - half)}, "explain before the published range"),
    ]
    for split in SERVING_SPLITS:
        m = truth.matrices[("demand_point", split)]
        have = np.isfinite(m.by_column["point"])
        empty = np.flatnonzero(~have.any(axis=1))
        partial = np.flatnonzero(have.any(axis=1) & ~have.all(axis=1))
        if len(empty):
            t = m.times[int(empty[0])]
            rows.append(
                (EXPLAIN, {"issue_time": iso(t)}, f"explain where nothing is stored ({split})")
            )
        if len(partial):
            k = int(partial[0])
            h = int(np.flatnonzero(~have[k])[0]) + 1
            rows.append(
                (
                    EXPLAIN,
                    {"issue_time": iso(m.times[k]), "lead": str(h)},
                    f"explain a lead without a stored forecast ({split})",
                )
            )
    if truth.actuals_first_end is not None:
        hi = truth.actuals_first_end - pd.Timedelta(minutes=truth.step)
        rows.append(
            (ACTUALS, {"start": iso(hi - pd.Timedelta(days=1)), "end": iso(hi)}, "before actuals")
        )
    if truth.other_event is not None:
        rows.append((EVENT.format(truth.other_event), {}, "an event that is not served (v1)"))
    return rows


def search_base(truth: Truth, issue_time: pd.Timestamp) -> dict[str, dict[str, str]]:
    """Each searched route's other parameters (valid), for the published data."""
    day = truth.actual_days[0]
    return {
        ACTUALS: {"start": iso(day), "end": iso(day + pd.Timedelta(days=1))},
        ANOMALIES: dict(EVERY_EVENT),
        RECOMPUTE: {"issue_time": iso(issue_time)},
        EXPLAIN: {"issue_time": iso(issue_time)},
    }


def run_search(
    client: httpx.Client,
    base: dict[str, dict[str, str]],
    parameters: Sequence[tuple[str, str]] = SEARCH_PARAMS,
    examples: int = SEARCH_EXAMPLES,
    allowed: tuple[int, ...] = (200, 404, 422),
) -> Tally:
    """The suites' hypothesis search over query parameters (derandomised): every answer is
    ``allowed`` (no 500), and an error has ``{"detail": <message>}``."""
    from hypothesis import HealthCheck, Phase, given, settings
    from hypothesis import strategies as st

    tally = Tally("P6", "no-500 search")
    statuses: dict[str, int] = {}

    @settings(
        max_examples=examples,
        deadline=None,
        derandomize=True,
        database=None,
        phases=(Phase.explicit, Phase.generate),
        suppress_health_check=list(HealthCheck),
    )
    @given(
        path=st.sampled_from(list(parameters)),
        value=st.one_of(st.text(max_size=40), st.sampled_from(SEARCH_LITERALS)),
    )
    def search(path: tuple[str, str], value: str) -> None:
        route, name = path
        tally.requests += 1
        try:
            r = client.get(route, params={**base.get(route, {}), name: value})
        except httpx.HTTPError as exc:
            tally.bad_responses += 1
            tally.note(f"{route} {name}={value!r}: {type(exc).__name__}")
            return
        statuses[str(r.status_code)] = statuses.get(str(r.status_code), 0) + 1
        ok = r.status_code in allowed
        if ok and r.status_code != 200:
            try:
                ok = _error_body(r.json())
            except ValueError:
                ok = False
        tally.check(ok, f"{route} {name}={value!r}: HTTP {r.status_code} {r.text[:120]}")

    search()
    tally.extra.update(statuses=dict(sorted(statuses.items())), examples=examples)
    return tally


# --------------------------------------------------------------------------- P5


@dataclass(frozen=True, slots=True)
class Request:
    endpoint: str
    path: str
    params: dict[str, str]


def _month_starts(truth: Truth) -> list[pd.Timestamp]:
    """NEM calendar months from the first served event's to the last's (UTC starts)."""
    if not truth.anomalies:
        return []
    ends = [pd.Timestamp(r["first_interval_end"]) for r in truth.anomalies]
    nem = [t.tz_convert(NEM_TZ) for t in (min(ends), max(ends))]
    months = pd.period_range(nem[0].strftime("%Y-%m"), nem[1].strftime("%Y-%m"), freq="M")
    return [nem_midnight_utc(str(p.start_time.date())) for p in months] + [
        nem_midnight_utc(str((months[-1] + 1).start_time.date()))
    ]


def p5_requests(
    truth: Truth, vcfg: ServingVerifyConfig, endpoints: Sequence[str]
) -> dict[str, list[Request]]:
    """Each endpoint's warm-up + timed requests, drawn uniformly (with replacement)."""
    out = {}
    first, n_targets = evolution_domain(truth)
    months = _month_starts(truth)
    stored = truth.stored_times("dev").append(truth.stored_times("holdout"))
    price = truth.published("price_persistence").intersection(
        truth.published("price_median_profile")
    )
    for name in endpoints:
        n = vcfg.warmup_requests + (
            vcfg.model_requests if name in MODEL_ENDPOINTS else vcfg.stored_requests
        )
        rng = _stream(vcfg.seed, f"p5:{name}")
        reqs: list[Request]
        if name == "forecast":
            times = truth.published("demand_point")
            reqs = [
                Request(name, DEMAND, {"issue_time": iso(times[int(k)])})
                for k in rng.integers(len(times), size=n)
            ]
        elif name == "price":
            reqs = [
                Request(name, PRICE, {"issue_time": iso(price[int(k)])})
                for k in rng.integers(len(price), size=n)
            ]
        elif name in MODEL_ENDPOINTS:
            path = RECOMPUTE if name == "recompute" else EXPLAIN
            reqs = [
                Request(name, path, {"issue_time": iso(stored[int(k)])})
                for k in rng.integers(len(stored), size=n)
            ]
        elif name == "evolution":
            reqs = [
                Request(
                    name,
                    EVOLUTION,
                    {"target_end": iso(first + pd.Timedelta(minutes=truth.step * (int(k) + 1)))},
                )
                for k in rng.integers(n_targets, size=n)
            ]
        elif name == "actuals":
            days = truth.actual_days
            reqs = []
            for k in rng.integers(len(days), size=n):
                lo = days[int(k)]
                reqs.append(
                    Request(
                        name, ACTUALS, {"start": iso(lo), "end": iso(lo + pd.Timedelta(days=1))}
                    )
                )
        elif name == "anomalies":
            reqs = []
            for k in rng.integers(len(months) - 1, size=n):
                lo, hi = months[int(k)], months[int(k) + 1]
                reqs.append(
                    Request(
                        name,
                        ANOMALIES,
                        {"start": iso(lo), "end": iso(hi), "limit": str(ANOMALY_LIMIT)},
                    )
                )
        elif name == "event":
            ids = [int(r["event_id"]) for r in truth.anomalies]
            reqs = [
                Request(name, EVENT.format(ids[int(k)]), {}) for k in rng.integers(len(ids), size=n)
            ]
        else:
            raise ValueError(f"unknown P5 endpoint {name!r}")
        out[name] = reqs
    return out


def p5_order(
    requests: dict[str, list[Request]], warmups: int, seed: int
) -> tuple[list[Request], list[Request]]:
    """The warm-ups (each endpoint's first ``warmups``) and the timed requests, each list in
    one seeded random order across endpoints (interleaved, step 5's question round)."""
    warm = [r for reqs in requests.values() for r in reqs[:warmups]]
    timed = [r for reqs in requests.values() for r in reqs[warmups:]]
    warm = [warm[int(i)] for i in _stream(seed, "p5:warmup_order").permutation(len(warm))]
    timed = [timed[int(i)] for i in _stream(seed, "p5:order").permutation(len(timed))]
    return warm, timed


def _timed_get(client: httpx.Client, req: Request) -> tuple[float, int, int]:
    """Milliseconds from sending the request to the last byte of its body; status; bytes."""
    t0 = time.perf_counter()
    try:
        r = client.get(req.path, params=req.params or None)
    except httpx.HTTPError:
        return (time.perf_counter() - t0) * 1000, 0, 0
    return (time.perf_counter() - t0) * 1000, r.status_code, len(r.content)


def latency_stats(ms: Sequence[float], statuses: Sequence[int], limit: float) -> dict[str, Any]:
    a = np.asarray(ms, dtype=np.float64)
    p50, p95, p99 = (float(v) for v in np.percentile(a, [50, 95, 99]))
    non_200 = int(sum(s != 200 for s in statuses))
    return {
        "n": len(a),
        "non_200": non_200,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": float(a.max()),
        "mean_ms": float(a.mean()),
        "limit_ms": limit,
        "passed": non_200 == 0 and p95 <= limit,
    }


def run_p5(
    client: httpx.Client,
    warm: Sequence[Request],
    timed: Sequence[Request],
    limits: dict[str, float],
) -> dict[str, Any]:
    """Warm-ups, then the timed requests, in the given order; each row of the logs is
    (endpoint, ms, status, bytes)."""
    warm_log = [(r.endpoint, *_timed_get(client, r)) for r in warm]
    timed_log = [(r.endpoint, *_timed_get(client, r)) for r in timed]
    endpoints = {}
    for name, limit in limits.items():
        rows = [x for x in timed_log if x[0] == name]
        if not rows:
            continue
        stats = latency_stats([ms for _, ms, _, _ in rows], [st for _, _, st, _ in rows], limit)
        stats["median_kib"] = float(np.median([b for _, _, _, b in rows])) / 1024
        endpoints[name] = stats
    return {
        "endpoints": endpoints,
        "passed": bool(endpoints) and all(e["passed"] for e in endpoints.values()),
        "warmup": warm_log,
        "timed": timed_log,
    }


# --------------------------------------------------------------------------- build, environment


def build_problems(health: Any, checkout: str | None) -> list[str]:
    """Why a verification run may not start (empty: it may). ``checkout`` is
    :func:`~gridcast.provenance.git_sha`'s ``<sha>[-dirty]``."""
    problems = []
    if checkout is None:
        problems.append("the checkout's commit is unknown (not a git checkout?)")
    elif checkout.endswith("-dirty"):
        problems.append("the checkout has uncommitted changes to tracked files")
    if not isinstance(health, dict):
        return [*problems, "no /health body"]
    if health.get("status") != "ok":
        failing = [k for k, c in (health.get("checks") or {}).items() if not c.get("ok")]
        problems.append(f"/health is {health.get('status')!r} (failing: {', '.join(failing)})")
    build = health.get("build") or {}
    if build.get("source") != "image":
        problems.append(
            f"the API is not the container image (build source {build.get('source')!r})"
        )
    if build.get("dirty") is not False:
        problems.append(f"the image is not clean (dirty: {build.get('dirty')!r})")
    head = checkout.removesuffix("-dirty") if checkout else None
    if head is not None and build.get("git_sha") != head:
        problems.append(
            f"the image is at {str(build.get('git_sha'))[:12]}, the checkout at {head[:12]}: "
            "run make api-up"
        )
    return problems


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=True, timeout=60
    ).stdout


def docker_environment() -> dict[str, Any]:
    """The Docker VM and the API image (``docker info`` / ``inspect``); errors are recorded."""
    out: dict[str, Any] = {}
    try:
        info = json.loads(_docker("info", "--format", "{{json .}}"))
        out["vm"] = {
            k: info.get(k)
            for k in (
                "NCPU",
                "MemTotal",
                "ServerVersion",
                "OperatingSystem",
                "Architecture",
                "KernelVersion",
            )
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        out["vm"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        container = json.loads(_docker("inspect", CONTAINERS[0]))[0]
        image = json.loads(_docker("image", "inspect", container["Image"]))[0]
        out["api_container"] = {
            "image_id": container["Image"],
            "image_size_bytes": image.get("Size"),
            "started_at": container.get("State", {}).get("StartedAt"),
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        out["api_container"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def docker_memory() -> dict[str, Any]:
    """``docker stats`` of the API and database containers, once (reported only)."""
    try:
        lines = _docker("stats", "--no-stream", "--format", "{{json .}}", *CONTAINERS)
        stats = [json.loads(line) for line in lines.splitlines() if line.strip()]
        return {
            s["Name"]: {"mem_usage": s.get("MemUsage"), "mem_perc": s.get("MemPerc")} for s in stats
        }
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def point_outside_50(engine: Engine, truth: Truth) -> dict[str, dict[str, Any]]:
    """Per split, the published pairs whose point lies outside the 50% interval (SQL over
    every pair; reported only, plan §3.1: CQR is not centred on the point)."""
    out = {}
    with engine.connect() as conn:
        for split in SERVING_SPLITS:
            pp, ip = truth.pubs[("demand_point", split)], truth.pubs[("demand_interval", split)]
            r = conn.execute(
                text(
                    """
                    SELECT count(*) AS pairs,
                           count(*) FILTER (WHERE f.point < i.lo50) AS below,
                           count(*) FILTER (WHERE f.point > i.hi50) AS above
                    FROM forecasts f
                    JOIN forecast_intervals i ON i.run_id = :ir AND i.issue_time = f.issue_time
                         AND i.target_end = f.target_end
                    WHERE f.run_id = :pr AND f.issue_time BETWEEN :lo AND :hi
                    """
                ),
                {
                    "pr": pp.run_id,
                    "ir": ip.run_id,
                    "lo": pp.first.to_pydatetime(),
                    "hi": pp.last.to_pydatetime(),
                },
            ).one()
            pairs = int(r.pairs)
            outside = int(r.below) + int(r.above)
            out[split] = {
                "pairs": pairs,
                "below": int(r.below),
                "above": int(r.above),
                "outside": outside,
                "share": outside / pairs if pairs else None,
            }
    return out


# --------------------------------------------------------------------------- the run


@dataclass(slots=True)
class VerifyResult:
    url: str
    started_at: str
    checkout_git_sha: str
    build: dict[str, Any]
    settings: dict[str, Any]
    environment: dict[str, Any]
    samples: dict[str, Any]
    checks: list[Tally]
    p5: dict[str, Any]
    reported: dict[str, Any]
    build_after: dict[str, Any] | None = None
    finished_at: str | None = None
    seconds: dict[str, float] = field(default_factory=dict)

    def verdicts(self) -> dict[str, bool]:
        out = {}
        for c in ("P2", "P3", "P6"):
            mine = [t for t in self.checks if t.criterion == c]
            out[c] = bool(mine) and all(t.passed for t in mine)
        out["P5"] = bool(self.p5.get("passed"))
        return out

    @property
    def build_unchanged(self) -> bool:
        return self.build_after == self.build

    @property
    def passed(self) -> bool:
        return self.build_unchanged and all(self.verdicts().values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "checkout_git_sha": self.checkout_git_sha,
            "build": self.build,
            "build_after": self.build_after,
            "build_unchanged": self.build_unchanged,
            "verdicts": self.verdicts(),
            "passed": self.passed,
            "settings": self.settings,
            "environment": self.environment,
            "samples": self.samples,
            "checks": [t.as_dict() for t in self.checks],
            "p5": self.p5,
            "reported": self.reported,
            "seconds": self.seconds,
        }


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_verify(
    client: httpx.Client,
    engine: Engine,
    cfg: AppConfig,
    checkout: str | None,
    url: str,
    committed_schema: dict[str, Any],
    truth: Truth | None = None,
    docker: bool = True,
) -> VerifyResult:
    """The whole verification (module docstring); :class:`VerifyRefused` before measuring."""
    started = _now()
    probe = Tally("-", "health")
    health = fetch(client, probe, "/health")
    problems = build_problems(health, checkout)
    if problems:
        raise VerifyRefused("; ".join(problems))
    assert isinstance(health, dict) and checkout is not None
    vcfg = cfg.serving.verify
    seconds: dict[str, float] = {}
    t0 = time.perf_counter()
    truth = truth if truth is not None else load_truth(engine, cfg)
    samples = draw_samples(truth, vcfg)
    seconds["truth"] = time.perf_counter() - t0
    log.info("api_verify.truth", seconds=round(seconds["truth"], 1), folds=len(truth.folds))
    environment = docker_environment() if docker else {}
    times = samples.issue_times
    tol = cfg.anomalies.explain.sum_tolerance

    def timed(fn: Callable[[], Tally]) -> Tally:
        t = time.perf_counter()
        tally = fn()
        tally.seconds = time.perf_counter() - t
        log.info(
            "api_verify.check",
            criterion=tally.criterion,
            check=tally.name,
            passed=tally.passed,
            requests=tally.requests,
            mismatches=tally.mismatches,
            seconds=round(tally.seconds, 1),
        )
        return tally

    checks = [
        timed(lambda: check_schema(client, committed_schema)),
        timed(lambda: check_errors(client, "invalid inputs (422)", INVALID_INPUTS, 422)),
        timed(
            lambda: check_errors(client, "nothing published (404)", not_published_table(truth), 404)
        ),
        timed(lambda: run_search(client, search_base(truth, times[0]))),
        timed(lambda: check_demand(client, truth, times)),
        timed(lambda: check_price(client, truth, times)),
        timed(lambda: check_evolution(client, truth, samples.targets)),
        timed(lambda: check_actuals(client, truth, samples.days)),
        timed(lambda: check_anomalies(client, truth)),
        timed(lambda: check_recompute(client, truth, times)),
        timed(lambda: check_explain(client, truth, times, tol)),
    ]
    t = time.perf_counter()
    endpoints = (*STORED_ENDPOINTS, *MODEL_ENDPOINTS)
    warm, timed_reqs = p5_order(
        p5_requests(truth, vcfg, endpoints), vcfg.warmup_requests, vcfg.seed
    )
    limits = {
        e: vcfg.model_p95_ms if e in MODEL_ENDPOINTS else vcfg.stored_p95_ms for e in endpoints
    }
    p5 = run_p5(client, warm, timed_reqs, limits)
    seconds["p5"] = time.perf_counter() - t
    log.info("api_verify.p5", passed=p5["passed"], seconds=round(seconds["p5"], 1))
    reported = {
        "memory_after": docker_memory() if docker else {},
        "point_outside_50": point_outside_50(engine, truth),
    }
    after = fetch(client, probe, "/health")
    result = VerifyResult(
        url=url,
        started_at=started,
        checkout_git_sha=checkout,
        build=health.get("build") or {},
        settings={**vcfg.model_dump(), "sum_tolerance": tol, "max_examples": MAX_EXAMPLES},
        environment=environment,
        samples=samples.as_dict(),
        checks=checks,
        p5=p5,
        reported=reported,
        build_after=(after or {}).get("build") if isinstance(after, dict) else None,
        finished_at=_now(),
        seconds={**seconds, **{f"{c.criterion} {c.name}": c.seconds for c in checks}},
    )
    return result


def save(result: VerifyResult, cfg: AppConfig) -> Path:
    """``<data root>/serving/verify_<UTC time>.json``; refuses to overwrite a stored run."""
    stamp = result.started_at.replace("-", "").replace(":", "").replace("T", "_").rstrip("Z")
    path = cfg.data_path(Path(cfg.serving.output_dir)) / f"verify_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(result.as_dict(), fh, indent=1, default=str)
        fh.write("\n")
    return path


def render(result: VerifyResult) -> str:
    """A terminal summary (the report is ``gridcast report-m5``)."""
    v = result.verdicts()
    lines = [
        f"api-verify {result.url} at {result.started_at} (image {result.build.get('git_sha')}): "
        + ("PASS" if result.passed else "FAIL"),
        "  "
        + ", ".join(f"{k} {'pass' if ok else 'FAIL'}" for k, ok in v.items())
        + ("" if result.build_unchanged else "; the API's build changed during the run"),
    ]
    for t in result.checks:
        lines.append(
            f"  {'ok  ' if t.passed else 'FAIL'} {t.criterion} {t.name:<26} "
            f"{t.requests:>6} requests, {t.values_compared:>9,} values, "
            f"{t.fields_compared:>9,} fields, {t.mismatches} mismatches, "
            f"{t.bad_responses} bad responses ({t.seconds:.0f} s)"
        )
        lines += [f"         {e}" for e in t.examples[:3]]
    for name, e in result.p5.get("endpoints", {}).items():
        lines.append(
            f"  {'ok  ' if e['passed'] else 'FAIL'} P5 {name:<10} p50 {e['p50_ms']:.1f} ms, "
            f"p95 {e['p95_ms']:.1f} ms (≤ {e['limit_ms']:g}), p99 {e['p99_ms']:.1f} ms, "
            f"{e['non_200']} not 200"
        )
    return "\n".join(lines)
