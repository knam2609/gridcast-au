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


@app.command()
def dq(start: StartOpt = None, end: EndOpt = None, regions: RegionsOpt = None) -> None:
    """Run data-quality checks over the configured window and store results."""
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


if __name__ == "__main__":
    app()
