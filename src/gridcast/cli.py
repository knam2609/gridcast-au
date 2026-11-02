"""Command-line entry point: ``uv run gridcast --help``."""

from __future__ import annotations

import typer

from gridcast import __version__
from gridcast.config import get_settings, load_config
from gridcast.logs import configure_logging, get_logger

app = typer.Typer(
    name="gridcast",
    help="GridCast-AU: NEM electricity data pipeline and forecasting.",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger(__name__)


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


if __name__ == "__main__":
    app()
