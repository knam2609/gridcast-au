"""Typed response models of every endpoint (m5_plan.md §2.4; P6: every endpoint has one).

Times are UTC. Float values are the stored ones: points as the float64 of their float32,
bounds as float32 read back as float64, so a client casting a value to float32 recovers the
stored bits. ``None`` (JSON ``null``) marks a pair without a stored value.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field

Split = Literal["dev", "holdout"]


class ErrorBody(BaseModel):
    """Every error response: 404 (nothing published), 422 (invalid input), 503 (unavailable)."""

    detail: str


# --------------------------------------------------------------------------- forecasts


class DemandLead(BaseModel):
    lead: int = Field(description="Lead in 5-minute steps, 1…288")
    lead_minutes: int
    target_end: dt.datetime = Field(description="End of the 5-minute target interval (UTC)")
    point_mw: float | None = Field(description="LightGBM C point forecast")
    lo95_mw: float | None
    lo80_mw: float | None
    lo50_mw: float | None
    hi50_mw: float | None
    hi80_mw: float | None
    hi95_mw: float | None


class DemandForecast(BaseModel):
    region: str
    issue_time: dt.datetime
    split: Split
    unit: Literal["MW"] = "MW"
    point_run_id: int
    interval_run_id: int
    interval_method: str
    leads: list[DemandLead]


class EvolutionItem(BaseModel):
    issue_time: dt.datetime
    split: Split
    point_run_id: int
    interval_run_id: int
    lead: int
    lead_minutes: int
    point_mw: float | None
    lo95_mw: float | None
    lo80_mw: float | None
    lo50_mw: float | None
    hi50_mw: float | None
    hi80_mw: float | None
    hi95_mw: float | None


class DemandEvolution(BaseModel):
    region: str
    target_end: dt.datetime
    unit: Literal["MW"] = "MW"
    items: list[EvolutionItem] = Field(description="Every published issue time in [T − 24 h, T)")


class PriceLead(BaseModel):
    lead: int
    lead_minutes: int
    target_end: dt.datetime
    persistence_aud_mwh: float | None
    median_profile_aud_mwh: float | None


class PriceForecast(BaseModel):
    region: str
    issue_time: dt.datetime
    split: Split
    unit: Literal["AUD/MWh"] = "AUD/MWh"
    baselines: Literal[True] = True
    label: str
    persistence_run_id: int
    median_profile_run_id: int
    leads: list[PriceLead]


# --------------------------------------------------------------------------- actuals


class ActualInterval(BaseModel):
    interval_start: dt.datetime
    interval_end: dt.datetime
    demand_mw: float | None
    price_aud_mwh: float | None
    price_status: str | None
    apc_flag_bits: int = Field(description="DISPATCHPRICE.APCFLAG, a 5-bit field")
    administered_price_binding: bool
    market_cap_floor_binding: bool
    price_manually_overwritten: bool
    market_suspended: bool


class Actuals(BaseModel):
    region: str
    start: dt.datetime
    end: dt.datetime
    intervals: list[ActualInterval] = Field(
        description="5-minute intervals lying inside [start, end]: start ≤ interval start and "
        "interval end ≤ end"
    )


# --------------------------------------------------------------------------- anomalies


class AnomalySummary(BaseModel):
    event_id: int
    run_id: int
    split: Split
    first_interval_end: dt.datetime
    last_interval_end: dt.datetime
    peak_interval_end: dt.datetime
    tier: Literal["notable", "severe"]
    detectors: list[str]
    categories: list[str]
    min_p: float | None
    lines_only: bool
    summary: str


class AnomalyEvent(AnomalySummary):
    evidence: dict[str, Any]


class AnomalyPage(BaseModel):
    region: str
    start: dt.datetime
    end: dt.datetime
    limit: int
    events: list[AnomalySummary] = Field(
        description="Events whose first interval end lies in [start, end), ordered by first "
        "interval end then event id"
    )
    next: str | None = Field(description="Pass as `after` for the next page; null at the end")


# --------------------------------------------------------------------------- health and meta


class HealthCheck(BaseModel):
    ok: bool
    detail: str


class Health(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    checks: dict[str, HealthCheck]
    demand_issue_times: tuple[dt.datetime, dt.datetime] | None
    price_issue_times: tuple[dt.datetime, dt.datetime] | None
    actuals: tuple[dt.datetime, dt.datetime] | None = Field(
        description="First and last stored interval end"
    )


class RunRef(BaseModel):
    run_id: int
    batch_id: str
    model_name: str
    git_sha: str | None
    config_hash: str | None


class MetricSet(BaseModel):
    source: str = Field(description="Where the values are stored (forecast_metrics row set)")
    n: int | None = Field(description="Scored pairs")
    values: dict[str, float | None]


class PublishedSplit(BaseModel):
    split: Split
    run: RunRef
    source_run: RunRef
    first_issue: dt.datetime
    last_issue: dt.datetime
    issue_times: int
    rows: int
    missing_pairs: int
    publication_id: int
    published_at: dt.datetime | None
    metrics: MetricSet | None


class Product(BaseModel):
    product: str
    label: str
    unit: str
    kind: Literal["point", "interval"]
    baseline: bool
    endpoint: str
    splits: list[PublishedSplit]


class AnomalyRun(BaseModel):
    split: Split
    run: RunRef
    recipe_version: int
    events: int
    first_event: dt.datetime | None
    last_event: dt.datetime | None


class Disclosure(BaseModel):
    key: str
    text: str
    sources: list[str]


class Licence(BaseModel):
    data: str
    licence: str
    attribution: str


class Splits(BaseModel):
    dev: tuple[dt.datetime, dt.datetime] = Field(description="Issue times [first, holdout start)")
    holdout: tuple[dt.datetime, dt.datetime] = Field(description="Issue times [start, end)")


class Meta(BaseModel):
    api_version: str
    region: str
    leads: int
    lead_minutes: int
    issue_every_minutes: int
    interval_levels: list[int]
    splits: Splits
    products: list[Product]
    anomalies: list[AnomalyRun]
    disclosures: list[Disclosure]
    licences: list[Licence]
