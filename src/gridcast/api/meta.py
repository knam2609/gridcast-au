"""``/v1/meta``: what is served, where it came from, how it scored, and its caveats.

The disclosures are plan §3.3's, one sentence each with the report it comes from (step 2
question round, 2026-11-23). They carry no numbers: every number ``/v1/meta`` shows comes from
``publications``, ``model_runs`` or ``forecast_metrics``. The licences follow the README's
data-sources table.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridcast.api import models as m
from gridcast.api.catalogue import Catalogue, RunInfo
from gridcast.config import SERVING_PRODUCTS, SERVING_SPLITS, AppConfig

API_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ProductInfo:
    label: str
    unit: str
    kind: str
    baseline: bool
    endpoint: str


PRODUCT_INFO: dict[str, ProductInfo] = {
    "demand_point": ProductInfo(
        "LightGBM C demand forecast (M2, the headline model): GFS temperature forecasts as "
        "published by the issue time, population-weighted over 5 Victorian sites",
        "MW",
        "point",
        False,
        "/v1/forecasts/demand",
    ),
    "demand_interval": ProductInfo(
        "Demand intervals at 50/80/95%: CQR around quantile LightGBM C (M3), rearranged",
        "MW",
        "interval",
        False,
        "/v1/forecasts/demand",
    ),
    "price_persistence": ProductInfo(
        "Price persistence (M1 baseline): the last known price for every lead",
        "AUD/MWh",
        "point",
        True,
        "/v1/forecasts/price",
    ),
    "price_median_profile": ProductInfo(
        "Price 28-day median profile (M1 baseline): the trailing 28-day median by day type "
        "(working or weekend/holiday) and local 5-minute slot",
        "AUD/MWh",
        "point",
        True,
        "/v1/forecasts/price",
    ),
}

DISCLOSURES: tuple[m.Disclosure, ...] = (
    m.Disclosure(
        key="intervals_rearranged",
        text=(
            "The served intervals are not M3's pre-registered headline but its reported-only "
            "'CQR, rearranged' row: the same bounds sorted per pair so the levels nest, a "
            "transform added after M3's step-3 results were seen."
        ),
        sources=["reports/m3_intervals.md", "reports/m3_holdout.md"],
    ),
    m.Disclosure(
        key="dev_forecasts_looked_at_twice",
        text=(
            "The development forecasts are M2's second development run; two runs looked at the "
            "same period, so development numbers are slightly optimistic, unlike the holdout's."
        ),
        sources=["reports/m2_demand.md", "reports/m2_holdout.md"],
    ),
    m.Disclosure(
        key="anomaly_v2_post_hoc",
        text=(
            "Anomaly recipe v2 was designed after v1's development verdicts, so its development "
            "verdicts are post hoc; the holdout year is its only blind test."
        ),
        sources=["reports/m4_anomalies_v2.md", "reports/m4_holdout.md"],
    ),
    m.Disclosure(
        key="price_baselines",
        text=(
            "Price forecasts are M1's baselines (persistence and the 28-day median profile), "
            "not a model, and have no intervals."
        ),
        sources=["reports/m1_baseline.md"],
    ),
    m.Disclosure(
        key="archive_not_live",
        text=(
            "This is an archive, not a live service: nothing issued after the holdout year is "
            "served (see `splits`)."
        ),
        sources=["docs/plans/m5_plan.md"],
    ),
)

LICENCES: tuple[m.Licence, ...] = (
    m.Licence(
        data="Market data (demand, price, flags) and every forecast's market inputs: AEMO MMSDM "
        "monthly archive and NEMWeb daily reports",
        licence="AEMO Copyright Permissions: use with attribution",
        attribution="Australian Energy Market Operator (AEMO)",
    ),
    m.Licence(
        data="Weather inputs of the LightGBM C forecasts: Open-Meteo Previous Runs (GFS) and "
        "Historical Weather (ERA5)",
        licence="CC BY 4.0 (non-commercial free tier)",
        attribution="Weather data by Open-Meteo.com, from NOAA GFS and ECMWF ERA5",
    ),
    m.Licence(
        data="Population weights of the weather sites",
        licence="CC BY 4.0",
        attribution="Australian Bureau of Statistics, Regional population 2024-25",
    ),
)


def run_ref(run: RunInfo) -> m.RunRef:
    return m.RunRef(
        run_id=run.run_id,
        batch_id=run.batch_id,
        model_name=run.model_name,
        git_sha=run.git_sha,
        config_hash=run.config_hash,
    )


def build_meta(cat: Catalogue, cfg: AppConfig) -> m.Meta:
    products = []
    for name in SERVING_PRODUCTS:
        info = PRODUCT_INFO[name]
        splits = []
        for split in SERVING_SPLITS:
            pub = cat.get(name, split)
            if pub is None:
                continue
            metrics = cat.metrics.get((pub.run_id, split))
            splits.append(
                m.PublishedSplit(
                    split=split,  # type: ignore[arg-type]
                    run=run_ref(cat.runs[pub.run_id]),
                    source_run=run_ref(cat.runs[pub.source_run_id]),
                    first_issue=pub.first_issue.to_pydatetime(),
                    last_issue=pub.last_issue.to_pydatetime(),
                    issue_times=pub.issue_times,
                    rows=pub.rows,
                    missing_pairs=pub.missing_pairs,
                    publication_id=pub.publication_id,
                    published_at=None
                    if pub.published_at is None
                    else pub.published_at.to_pydatetime(),
                    metrics=None
                    if metrics is None
                    else m.MetricSet(
                        source=f"forecast_metrics: run {pub.run_id}, split {split}, "
                        "segment all, all leads pooled",
                        n=metrics.n,
                        values=metrics.values,
                    ),
                )
            )
        products.append(
            m.Product(
                product=name,
                label=info.label,
                unit=info.unit,
                kind=info.kind,  # type: ignore[arg-type]
                baseline=info.baseline,
                endpoint=info.endpoint,
                splits=splits,
            )
        )
    anomalies = [
        m.AnomalyRun(
            split=a.split,  # type: ignore[arg-type]
            run=run_ref(a.run),
            recipe_version=int(a.run.config.get("recipe_version", 0)),
            events=a.events,
            first_event=None if a.first_event is None else a.first_event.to_pydatetime(),
            last_event=None if a.last_event is None else a.last_event.to_pydatetime(),
        )
        for a in cat.anomaly_runs.values()
    ]
    return m.Meta(
        api_version=API_VERSION,
        region=cat.region,
        leads=cfg.forecast.horizon_steps,
        lead_minutes=cfg.nem.dispatch_interval_minutes,
        issue_every_minutes=cfg.forecast.issue_every_minutes,
        interval_levels=sorted(cfg.serving.interval_levels),
        splits=m.Splits(
            dev=(cat.first_issue.to_pydatetime(), cat.holdout_start.to_pydatetime()),
            holdout=(cat.holdout_start.to_pydatetime(), cat.holdout_end.to_pydatetime()),
        ),
        products=products,
        anomalies=anomalies,
        disclosures=list(DISCLOSURES),
        licences=list(LICENCES),
    )
