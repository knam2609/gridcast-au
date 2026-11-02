"""Command-line entry point: ``uv run gridcast --help``."""

from __future__ import annotations

from typing import Annotated

import typer
from tabulate import tabulate

from gridcast import __version__
from gridcast.config import get_settings, load_config, load_market_limits
from gridcast.logs import configure_logging, get_logger

app = typer.Typer(
    name="gridcast",
    help="GridCast-AU: NEM electricity data pipeline and forecasting.",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger(__name__)

RegionsOpt = Annotated[
    str | None, typer.Option("--regions", help="Comma-separated region ids (default: config)")
]
StartOpt = Annotated[str | None, typer.Option("--start", help="First month YYYY-MM")]
EndOpt = Annotated[str | None, typer.Option("--end", help="Last month YYYY-MM")]


def _regions(value: str | None) -> list[str]:
    cfg = load_config()
    regions = [r.strip() for r in value.split(",")] if value else list(cfg.regions)
    for r in regions:
        cfg.region(r)
    return regions


@app.callback()
def _main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)


@app.command()
def info() -> None:
    """Show version and the resolved configuration."""
    cfg = load_config()
    typer.echo(f"gridcast {__version__}")
    typer.echo(f"regions: {', '.join(cfg.regions)}")
    typer.echo(f"ingestion window: {cfg.ingestion.start_month} → {cfg.ingestion.end_month}")
    typer.echo(
        f"forecast: issue every {cfg.forecast.issue_every_minutes} min, "
        f"{cfg.forecast.horizon_steps} x {cfg.nem.dispatch_interval_minutes}-min steps"
    )


@app.command()
def ingest(start: StartOpt = None, end: EndOpt = None, regions: RegionsOpt = None) -> None:
    """Download, validate and load AEMO MMSDM dispatch data month by month."""
    from gridcast.database.engine import get_engine
    from gridcast.ingestion.pipeline import ingest_range, parse_month_arg

    cfg = load_config()
    first = parse_month_arg(start or cfg.ingestion.start_month)
    last = parse_month_arg(end or cfg.ingestion.end_month)
    results = ingest_range(get_engine(), cfg, first, last, _regions(regions))
    table = [
        [f"{r.period:%Y-%m}", r.status, r.rows_parsed, r.duplicates_dropped, r.rows_loaded]
        for r in results
    ]
    typer.echo(
        tabulate(table, headers=["month", "status", "parsed", "dupes", "loaded"], tablefmt="github")
    )
    failed = [r for r in results if r.status != "success"]
    for r in failed:
        typer.echo(f"FAILED {r.period:%Y-%m}: {r.error}", err=True)
    if failed:
        raise typer.Exit(code=1)


@app.command("ingest-weather")
def ingest_weather(
    start: StartOpt = None,
    end: EndOpt = None,
    dataset: Annotated[
        str, typer.Option(help="forecasts (variant C), observations (variant D) or all")
    ] = "all",
    refresh: Annotated[
        bool, typer.Option(help="Re-download even when a cached response exists")
    ] = False,
) -> None:
    """Download, validate and load hourly Open-Meteo weather month by month."""
    from gridcast.database.engine import get_engine
    from gridcast.ingestion.pipeline import parse_month_arg
    from gridcast.ingestion.weather import Kind, ingest_weather_range

    choices: dict[str, tuple[Kind, ...]] = {
        "all": ("forecasts", "observations"),
        "forecasts": ("forecasts",),
        "observations": ("observations",),
    }
    if dataset not in choices:
        raise typer.BadParameter(f"dataset must be one of {sorted(choices)}")
    cfg = load_config()
    first = parse_month_arg(start or cfg.ingestion.start_month)
    last = parse_month_arg(end or cfg.ingestion.end_month)
    results = ingest_weather_range(
        get_engine(), cfg, first, last, choices[dataset], refresh=refresh
    )
    table = [
        [f"{r.period:%Y-%m}", r.dataset, r.status, r.rows_parsed, r.rows_loaded] for r in results
    ]
    typer.echo(
        tabulate(table, headers=["month", "dataset", "status", "rows", "loaded"], tablefmt="github")
    )
    failed = [r for r in results if r.status != "success"]
    for r in failed:
        typer.echo(f"FAILED {r.period:%Y-%m} {r.dataset}: {r.error}", err=True)
    if failed:
        raise typer.Exit(code=1)


@app.command()
def dq(start: StartOpt = None, end: EndOpt = None, regions: RegionsOpt = None) -> None:
    """Run market and weather data-quality checks over the window and store results."""
    from gridcast.database.engine import get_engine
    from gridcast.ingestion.pipeline import parse_month_arg
    from gridcast.validation.checks import run_checks

    cfg = load_config()
    results = run_checks(
        get_engine(),
        cfg,
        load_market_limits(cfg),
        _regions(regions),
        parse_month_arg(start or cfg.ingestion.start_month),
        parse_month_arg(end or cfg.ingestion.end_month),
    )
    table = [[r.check_name, r.region_id or "-", r.status.upper(), r.n_violations] for r in results]
    typer.echo(
        tabulate(table, headers=["check", "region", "status", "violations"], tablefmt="github")
    )
    if any(r.status == "fail" for r in results):
        raise typer.Exit(code=1)


@app.command()
def calendar(regions: RegionsOpt = None) -> None:
    """Build/refresh calendar_days (holidays, weekends, DST days) for the configured window.

    Whole calendar years are built (through the year after the window ends) so forecast
    horizons that extend past the last observed interval are covered.
    """
    import datetime as dt

    from gridcast.database.engine import get_engine
    from gridcast.database.load import ensure_regions
    from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
    from gridcast.ingestion.pipeline import parse_month_arg

    cfg = load_config()
    first = parse_month_arg(cfg.ingestion.start_month)
    last = parse_month_arg(cfg.ingestion.end_month)
    start, end = dt.date(first.year, 1, 1), dt.date(last.year + 1, 12, 31)
    with get_engine().begin() as conn:
        ensure_regions(conn, cfg)
        for region_id in _regions(regions):
            region = cfg.region(region_id)
            days = build_calendar_days(region.holiday_subdiv, region.local_tz, start, end)
            n = upsert_calendar_days(conn, days)
            typer.echo(
                f"{region.holiday_subdiv}: {n} days {start} → {end} "
                f"({int(days['is_public_holiday'].sum())} holidays, "
                f"{int(days['is_dst_change_day'].sum())} DST-change days)"
            )


TARGET_COLUMNS = {"demand": "total_demand_mw", "price": "rrp_aud_mwh"}


@app.command()
def backtest(
    target: Annotated[str, typer.Option(help="demand or price")] = "demand",
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    save_predictions: Annotated[
        bool, typer.Option(help="Write full backtest predictions to parquet")
    ] = True,
) -> None:
    """Walk-forward evaluation of the baseline models; results stored in PostgreSQL."""
    import time
    import uuid

    from gridcast.database.engine import get_engine
    from gridcast.database.queries import load_market_frame
    from gridcast.evaluation.store import save_predictions as save_parquet
    from gridcast.evaluation.store import store_backtest
    from gridcast.features.calendar import load_calendar_days
    from gridcast.forecasting.backtest import evaluate, run_backtest
    from gridcast.forecasting.origins import ForecastSetup, build_context, nem_midnight_utc
    from gridcast.models.baselines import default_baselines

    if target not in TARGET_COLUMNS:
        raise typer.BadParameter(f"target must be one of {sorted(TARGET_COLUMNS)}")
    cfg = load_config()
    region_cfg = cfg.region(region)
    setup = ForecastSetup.from_config(cfg)
    engine = get_engine()
    t0 = time.perf_counter()

    market = load_market_frame(engine, region)
    with engine.connect() as conn:
        calendar_days = load_calendar_days(conn, region_cfg.holiday_subdiv)
    ctx = build_context(market, TARGET_COLUMNS[target], region_cfg.local_tz, calendar_days)

    models = default_baselines(target, cfg.baselines.median_profile_lookback_days)
    start = nem_midnight_utc(str(cfg.backtest.eval_start))
    holdout_start = nem_midnight_utc(str(cfg.backtest.holdout_start))
    out = run_backtest(ctx, models, setup, target, start)
    metrics = evaluate(out, ctx, cfg.evaluation, holdout_start)

    batch_id = uuid.uuid4()
    artifacts = {}
    if save_predictions:
        written = save_parquet(out, cfg.data_path(cfg.paths.backtest_dir), batch_id)
        # Stored relative to the data root so the data can move without breaking references.
        artifacts = {name: path.relative_to(cfg.data_root()) for name, path in written.items()}
    run_config = {
        "setup": {
            "issue_every_steps": setup.issue_every_steps,
            "horizon_steps": setup.horizon_steps,
            "latency_steps": setup.latency_steps,
        },
        "eval_start": str(cfg.backtest.eval_start),
        "holdout_start": str(cfg.backtest.holdout_start),
        "n_origins": len(out.origins),
        "n_folds": len(out.folds),
        "scored_pairs": int(out.common_mask().sum()),
        "total_pairs": int(out.common_mask().size),
    }
    store_backtest(
        engine, out, metrics, models, region, holdout_start, run_config, artifacts, batch_id
    )

    pooled = metrics[(metrics["segment"] == "all") & (metrics["metric"] == "mae")]
    table = []
    for model in out.model_names:
        m = pooled[pooled["model"] == model].set_index(["split", "horizon"])["value"]
        table.append(
            [model]
            + [m.get(("dev", h)) for h in (1, 12, 48, 288)]
            + [m.get(("dev", 0)), m.get(("holdout", 0))]
        )
    unit = "MW" if target == "demand" else "$/MWh"
    typer.echo(f"\n{target} MAE ({unit}) — batch {batch_id} — {time.perf_counter() - t0:.0f}s")
    typer.echo(
        tabulate(
            table,
            headers=["model", "dev h=1", "dev h=12", "dev h=48", "dev h=288", "dev all", "holdout"],
            floatfmt=".1f",
            tablefmt="github",
        )
    )


@app.command()
def report(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write reports/m1_baseline.md and figures from the latest stored backtests."""
    from gridcast.database.engine import get_engine
    from gridcast.evaluation.report import write_report

    path = write_report(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
