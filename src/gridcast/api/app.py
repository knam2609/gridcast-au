"""The read-only GridCast-AU API (m5_plan.md §2.4).

Run: ``uvicorn gridcast.api.app:create_app --factory`` (``make api``).

* Every timestamp in and out is ISO 8601 in UTC; inputs need an explicit offset. An issue
  time must lie on the 30-minute grid and a target on the 5-minute grid.
* Errors are ``{"detail": "<message>"}``: 422 for invalid input, 404 for valid input with
  nothing published, 503 when the database (or a product) is unavailable. User input never
  produces a 500.
* ``/health`` answers 503 only when the stored endpoints cannot work (database down or a
  publication incomplete); ``degraded`` means they work but something else does not.
"""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, Path, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from gridcast.api import models as m
from gridcast.api import queries as q
from gridcast.api.catalogue import Catalogue, Published, latest_attempts, load_catalogue
from gridcast.api.catalogue import read_only_engine as make_engine
from gridcast.api.inputs import (
    MAX_ID,
    InputError,
    NotPublished,
    Unavailable,
    check_choice,
    check_grid,
    check_range,
    decode_cursor,
    encode_cursor,
    parse_instant,
)
from gridcast.api.meta import API_VERSION, build_meta
from gridcast.config import SERVING_PRODUCTS, SERVING_SPLITS, AppConfig, get_settings, load_config
from gridcast.logs import get_logger

log = get_logger(__name__)

ACTUALS_MAX_SPAN = pd.Timedelta(days=7)
DEFAULT_LIMIT = 100
MAX_LIMIT = 500
TIERS = ("notable", "severe")
CATEGORIES = (
    "unusual demand",
    "unexpected demand (short-term)",
    "unexpected demand (day-ahead)",
    "high price",
    "unusual price",
    "volatility change",
    "multivariate",
)
DETECTORS = (
    "demand_level",
    "demand_residual_short",
    "demand_residual_dayahead",
    "price_level",
    "price_volatility",
    "isolation_forest",
    "robust_distance",
    "price_line_300",
    "price_line_5000",
)
INTERVAL_METHOD = "CQR around quantile LightGBM C, bounds sorted per pair (rearranged)"
PRICE_LABEL = "Baselines, not a model: M1's persistence and 28-day median profile (no intervals)"
ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"model": m.ErrorBody, "description": "Valid input, nothing published there"},
    422: {"model": m.ErrorBody, "description": "Invalid input"},
    503: {"model": m.ErrorBody, "description": "The database or a product is unavailable"},
}


def description(cfg: AppConfig) -> str:
    last_day = cfg.backtest.holdout_end - dt.timedelta(days=1)  # type: ignore[operator]
    return (
        "Read-only archive of GridCast-AU's scored outputs for Victoria (VIC1): LightGBM C "
        "demand forecasts with CQR prediction intervals, the M1 price baselines, actuals and "
        f"anomaly events (recipe v2), for issue times {cfg.serving.first_issue:%-d %b %Y} → "
        f"{last_day:%-d %b %Y} (NEM time). Timestamps are ISO 8601 in UTC; inputs need an "
        "explicit offset (`Z`, or `%2B10:00` in a query string). See `/v1/meta` for "
        "provenance, scored metrics, disclosures and licences."
    )


RegionQ = Annotated[str, Query(description="NEM region; only VIC1 is published")]


class ApiState:
    """The configuration, the read-only engine and the catalogue (loaded once, lazily)."""

    def __init__(self, cfg: AppConfig, engine: Engine | None) -> None:
        self.cfg = cfg
        self._engine = engine
        self._catalogue: Catalogue | None = None
        self._lock = threading.RLock()  # catalogue() takes it, then engine does

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = make_engine(get_settings().database_url)
        return self._engine

    def catalogue(self) -> Catalogue:
        if self._catalogue is None:
            with self._lock:
                if self._catalogue is None:
                    try:
                        self._catalogue = load_catalogue(self.engine, self.cfg)
                    except OperationalError as exc:
                        raise Unavailable("database unavailable") from exc
        return self._catalogue


def _state(request: Request) -> ApiState:
    state: ApiState = request.app.state.api
    return state


def _need(cat: Catalogue, product: str, split: str) -> Published:
    pub = cat.get(product, split)
    if pub is None:
        raise Unavailable(f"{product}/{split} is not published")
    return pub


def _check_region(cat: Catalogue, region: str) -> None:
    if region != cat.region:
        raise NotPublished(f"region {region!r} is not published; only {cat.region!r} is")


def _range_text(pubs: list[Published]) -> str:
    return "; ".join(
        f"{p.split} {p.first_issue.isoformat()} → {p.last_issue.isoformat()}" for p in pubs
    )


def _bounds_fields(b: tuple[float, ...] | None) -> dict[str, float | None]:
    names = ("lo95_mw", "lo80_mw", "lo50_mw", "hi50_mw", "hi80_mw", "hi95_mw")
    return dict(zip(names, b if b is not None else (None,) * 6, strict=True))


def _anomaly_runs(cat: Catalogue) -> list[int]:
    if cat.anomaly_problems or set(cat.anomaly_runs) != set(SERVING_SPLITS):
        raise Unavailable("anomaly runs unavailable: " + "; ".join(cat.anomaly_problems))
    return cat.anomaly_run_ids


def _summary(r: Any) -> dict[str, Any]:
    return {
        "event_id": int(r.event_id),
        "run_id": int(r.run_id),
        "split": r.split,
        "first_interval_end": r.first_interval_end,
        "last_interval_end": r.last_interval_end,
        "peak_interval_end": r.peak_interval_end,
        "tier": r.tier,
        "detectors": list(r.detectors),
        "categories": list(r.categories),
        "min_p": None if r.min_p is None else float(r.min_p),
        "lines_only": bool(r.lines_only),
        "summary": r.summary,
    }


def _validation_message(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())[1:]) or "request"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "invalid request"


def _handler(status: int) -> Callable[[Request, Exception], Any]:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    return handle


def create_app(cfg: AppConfig | None = None, engine: Engine | None = None) -> FastAPI:
    """The API. ``engine`` defaults to a read-only engine on ``GRIDCAST_DATABASE_URL``."""
    state = ApiState(cfg or load_config(), engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            state.catalogue()
        except Unavailable as exc:  # the stored endpoints answer 503 until it loads
            log.warning("api.catalogue_unavailable", error=str(exc))
        yield

    app = FastAPI(
        title="GridCast-AU API",
        version=API_VERSION,
        description=description(state.cfg),
        lifespan=lifespan,
    )
    app.state.api = state
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])
    app.add_exception_handler(InputError, _handler(422))
    app.add_exception_handler(NotPublished, _handler(404))
    app.add_exception_handler(Unavailable, _handler(503))

    async def database_down(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "database unavailable"})

    async def invalid_request(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        return JSONResponse(status_code=422, content={"detail": _validation_message(exc)})

    app.add_exception_handler(OperationalError, database_down)
    app.add_exception_handler(RequestValidationError, invalid_request)

    cfg_ = state.cfg
    step = cfg_.nem.dispatch_interval_minutes
    issue_every = cfg_.forecast.issue_every_minutes
    n_leads = cfg_.forecast.horizon_steps

    # ------------------------------------------------------------------ health and meta

    @app.get(
        "/health",
        response_model=m.Health,
        responses={503: {"model": m.Health, "description": "Stored endpoints unavailable"}},
        tags=["service"],
    )
    def health(request: Request, response: Response) -> m.Health:
        """Database, publications (the latest attempt of each succeeded) and anomaly runs."""
        st = _state(request)
        checks: dict[str, m.HealthCheck] = {}
        cat: Catalogue | None = None
        try:
            q.ping(st.engine)
            checks["database"] = m.HealthCheck(ok=True, detail="reachable")
            cat = st.catalogue()
            latest = latest_attempts(st.engine, cat.region)
            bad = [
                f"{p}/{s}: {latest.get((p, s), 'never published')}"
                for p in SERVING_PRODUCTS
                for s in SERVING_SPLITS
                if latest.get((p, s)) != "success"
            ]
            checks["publications"] = m.HealthCheck(
                ok=not bad,
                detail="; ".join(bad) or "latest attempt of each of 8 succeeded",
            )
            problems = cat.anomaly_problems
            checks["anomaly_runs"] = m.HealthCheck(
                ok=not problems, detail="; ".join(problems) or "recipe v2, dev and holdout"
            )
        except (OperationalError, Unavailable) as exc:
            checks["database"] = m.HealthCheck(
                ok=False, detail=f"unavailable: {type(exc).__name__}"
            )
        core = ("database", "publications")
        if not all(k in checks and checks[k].ok for k in core):
            status = "unavailable"
            response.status_code = 503
        elif all(c.ok for c in checks.values()):
            status = "ok"
        else:
            status = "degraded"

        def pair(r: tuple[pd.Timestamp, pd.Timestamp] | None) -> Any:
            return None if r is None else (r[0].to_pydatetime(), r[1].to_pydatetime())

        return m.Health(
            status=status,  # type: ignore[arg-type]
            checks=checks,
            demand_issue_times=pair(cat.issue_range("demand_point")) if cat else None,
            price_issue_times=pair(cat.issue_range("price_persistence")) if cat else None,
            actuals=pair(cat.actuals) if cat else None,
        )

    @app.get("/v1/meta", response_model=m.Meta, responses={503: ERRORS[503]}, tags=["service"])
    def meta(request: Request) -> m.Meta:
        """What is served, its provenance and scored metrics, disclosures and licences."""
        st = _state(request)
        return build_meta(st.catalogue(), st.cfg)

    # ------------------------------------------------------------------ forecasts

    @app.get(
        "/v1/forecasts/demand",
        response_model=m.DemandForecast,
        responses=ERRORS,
        tags=["forecasts"],
    )
    def demand(
        request: Request,
        issue_time: Annotated[str, Query(description="Issue time on the 30-minute grid")],
        region: RegionQ = "VIC1",
    ) -> m.DemandForecast:
        """All 288 leads of one issue time: LightGBM C's point and the 50/80/95% bounds
        (``null`` where nothing was stored)."""
        t = check_grid("issue_time", parse_instant("issue_time", issue_time), issue_every)
        st = _state(request)
        cat = st.catalogue()
        _check_region(cat, region)
        split = cat.split_of(t)
        point_pub = _need(cat, "demand_point", split)
        interval_pub = _need(cat, "demand_interval", split)
        if not point_pub.covers(t):
            pubs = [p for s in SERVING_SPLITS if (p := cat.get("demand_point", s))]
            raise NotPublished(
                f"no demand forecast issued at {t.isoformat()}; published: {_range_text(pubs)}"
            )
        pts = q.points(st.engine, point_pub.run_id, t)
        bnd = q.bounds(st.engine, interval_pub.run_id, t) if interval_pub.covers(t) else {}
        leads = [
            m.DemandLead(
                lead=h,
                lead_minutes=h * step,
                target_end=(t + pd.Timedelta(minutes=h * step)).to_pydatetime(),
                point_mw=pts.get(h),
                **_bounds_fields(bnd.get(h)),
            )
            for h in range(1, n_leads + 1)
        ]
        return m.DemandForecast(
            region=cat.region,
            issue_time=t.to_pydatetime(),
            split=split,  # type: ignore[arg-type]
            point_run_id=point_pub.run_id,
            interval_run_id=interval_pub.run_id,
            interval_method=INTERVAL_METHOD,
            leads=leads,
        )

    @app.get(
        "/v1/forecasts/demand/evolution",
        response_model=m.DemandEvolution,
        responses=ERRORS,
        tags=["forecasts"],
    )
    def evolution(
        request: Request,
        target_end: Annotated[str, Query(description="Target interval end, 5-minute grid")],
        region: RegionQ = "VIC1",
    ) -> m.DemandEvolution:
        """Every forecast published for one target: the issue times in [T − 24 h, T)."""
        target = check_grid("target_end", parse_instant("target_end", target_end), step)
        st = _state(request)
        cat = st.catalogue()
        _check_region(cat, region)
        lo = target - pd.Timedelta(minutes=n_leads * step)
        grid = pd.Timedelta(minutes=issue_every)
        first = lo + (-(lo - pd.Timestamp(0, tz="UTC")) % grid)
        issues = []
        for t in pd.date_range(first, target - pd.Timedelta(microseconds=1), freq=grid):
            pub = cat.get("demand_point", cat.split_of(t))
            if pub is not None and pub.covers(t):
                issues.append(t.as_unit("us"))
        if not issues:
            raise NotPublished(f"no published forecast targets {target.isoformat()}")
        splits = sorted({cat.split_of(t) for t in issues})
        point_runs = {s: _need(cat, "demand_point", s).run_id for s in splits}
        interval_runs = {s: _need(cat, "demand_interval", s).run_id for s in splits}
        pts, bnd = q.evolution(
            st.engine, list(point_runs.values()), list(interval_runs.values()), issues[0], target
        )
        items = []
        for t in issues:
            split = cat.split_of(t)
            lead = int((target - t) / pd.Timedelta(minutes=step))
            items.append(
                m.EvolutionItem(
                    issue_time=t.to_pydatetime(),
                    split=split,  # type: ignore[arg-type]
                    point_run_id=point_runs[split],
                    interval_run_id=interval_runs[split],
                    lead=lead,
                    lead_minutes=lead * step,
                    point_mw=pts.get(t),
                    **_bounds_fields(bnd.get(t)),
                )
            )
        return m.DemandEvolution(region=cat.region, target_end=target.to_pydatetime(), items=items)

    @app.get(
        "/v1/forecasts/price",
        response_model=m.PriceForecast,
        responses=ERRORS,
        tags=["forecasts"],
    )
    def price(
        request: Request,
        issue_time: Annotated[str, Query(description="Issue time on the 30-minute grid")],
        region: RegionQ = "VIC1",
    ) -> m.PriceForecast:
        """All 288 leads of both M1 price baselines for one issue time (labelled baselines)."""
        t = check_grid("issue_time", parse_instant("issue_time", issue_time), issue_every)
        st = _state(request)
        cat = st.catalogue()
        _check_region(cat, region)
        split = cat.split_of(t)
        pers = _need(cat, "price_persistence", split)
        prof = _need(cat, "price_median_profile", split)
        if not (pers.covers(t) and prof.covers(t)):
            pubs = [p for s in SERVING_SPLITS if (p := cat.get("price_persistence", s))]
            raise NotPublished(
                f"no price baseline issued at {t.isoformat()}; published: {_range_text(pubs)}"
            )
        a = q.points(st.engine, pers.run_id, t)
        b = q.points(st.engine, prof.run_id, t)
        leads = [
            m.PriceLead(
                lead=h,
                lead_minutes=h * step,
                target_end=(t + pd.Timedelta(minutes=h * step)).to_pydatetime(),
                persistence_aud_mwh=a.get(h),
                median_profile_aud_mwh=b.get(h),
            )
            for h in range(1, n_leads + 1)
        ]
        return m.PriceForecast(
            region=cat.region,
            issue_time=t.to_pydatetime(),
            split=split,  # type: ignore[arg-type]
            label=PRICE_LABEL,
            persistence_run_id=pers.run_id,
            median_profile_run_id=prof.run_id,
            leads=leads,
        )

    # ------------------------------------------------------------------ actuals

    @app.get("/v1/actuals", response_model=m.Actuals, responses=ERRORS, tags=["actuals"])
    def actuals(
        request: Request,
        start: Annotated[str, Query(description="Window start (inclusive)")],
        end: Annotated[str, Query(description="Window end (inclusive); at most 7 days")],
        region: RegionQ = "VIC1",
    ) -> m.Actuals:
        """5-minute demand, price and market flags of the intervals inside [start, end]."""
        lo, hi = parse_instant("start", start), parse_instant("end", end)
        check_range(lo, hi, ACTUALS_MAX_SPAN)
        st = _state(request)
        cat = st.catalogue()
        _check_region(cat, region)
        rows = q.actuals(st.engine, cat.region, lo, hi)
        if not rows:
            window = (
                "none stored"
                if cat.actuals is None
                else f"{cat.actuals[0].isoformat()} → {cat.actuals[1].isoformat()}"
            )
            raise NotPublished(f"no interval lies inside the window; stored: {window}")
        return m.Actuals(
            region=cat.region,
            start=lo.to_pydatetime(),
            end=hi.to_pydatetime(),
            intervals=[
                m.ActualInterval(
                    interval_start=r.interval_start,
                    interval_end=r.interval_end,
                    demand_mw=r.total_demand_mw,
                    price_aud_mwh=r.rrp_aud_mwh,
                    price_status=r.price_status,
                    apc_flag_bits=int(r.apc_flag_bits),
                    administered_price_binding=bool(r.administered_price_binding),
                    market_cap_floor_binding=bool(r.market_cap_floor_binding),
                    price_manually_overwritten=bool(r.price_manually_overwritten),
                    market_suspended=bool(r.market_suspended),
                )
                for r in rows
            ],
        )

    # ------------------------------------------------------------------ anomalies

    @app.get("/v1/anomalies", response_model=m.AnomalyPage, responses=ERRORS, tags=["anomalies"])
    def anomalies(
        request: Request,
        start: Annotated[str, Query(description="First interval end from (inclusive)")],
        end: Annotated[str, Query(description="First interval end before (exclusive)")],
        tier: Annotated[str | None, Query(description=f"One of {list(TIERS)}")] = None,
        category: Annotated[str | None, Query(description="An event category")] = None,
        detector: Annotated[str | None, Query(description="A detector that fired")] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
        after: Annotated[str | None, Query(description="`next` of the previous page")] = None,
        region: RegionQ = "VIC1",
    ) -> m.AnomalyPage:
        """Anomaly events (recipe v2) without evidence, one page at a time."""
        lo, hi = parse_instant("start", start), parse_instant("end", end)
        check_range(lo, hi, None)
        check_choice("tier", tier, TIERS)
        check_choice("category", category, CATEGORIES)
        check_choice("detector", detector, DETECTORS)
        position = decode_cursor(after) if after is not None else None
        st = _state(request)
        cat = st.catalogue()
        _check_region(cat, region)
        runs = _anomaly_runs(cat)
        rows = q.anomaly_page(
            st.engine, runs, lo, hi, limit + 1, tier, category, detector, position
        )
        more = len(rows) > limit
        rows = rows[:limit]
        last = rows[-1] if rows else None
        return m.AnomalyPage(
            region=cat.region,
            start=lo.to_pydatetime(),
            end=hi.to_pydatetime(),
            limit=limit,
            events=[m.AnomalySummary(**_summary(r)) for r in rows],
            next=encode_cursor(pd.Timestamp(last.first_interval_end), int(last.event_id))
            if more and last is not None
            else None,
        )

    @app.get(
        "/v1/anomalies/{event_id}",
        response_model=m.AnomalyEvent,
        responses=ERRORS,
        tags=["anomalies"],
    )
    def anomaly(
        request: Request, event_id: Annotated[int, Path(ge=1, le=MAX_ID)]
    ) -> m.AnomalyEvent:
        """One served event with its full evidence."""
        st = _state(request)
        runs = _anomaly_runs(st.catalogue())
        row = q.anomaly_event(st.engine, runs, event_id)
        if row is None:
            raise NotPublished(f"no served anomaly event {event_id} (recipe v2 only)")
        return m.AnomalyEvent(**_summary(row), evidence=dict(row.evidence))

    return app
