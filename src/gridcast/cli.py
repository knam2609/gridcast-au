"""Command-line entry point: ``uv run gridcast --help``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from tabulate import tabulate

from gridcast import __version__
from gridcast.config import get_settings, load_config, load_market_limits
from gridcast.logs import configure_logging, get_logger

if TYPE_CHECKING:
    from gridcast.ingestion.daily import MatchCheck

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


@app.command("ingest-daily")
def ingest_daily_cmd(
    day: Annotated[
        list[str], typer.Option("--day", help="NEM-time day YYYY-MM-DD to load (repeatable)")
    ],
    check_month: Annotated[
        str,
        typer.Option(
            "--check-month", help="Month YYYY-MM loaded from MMSDM, compared with the daily reports"
        ),
    ],
    regions: RegionsOpt = None,
) -> None:
    """Load days MMSDM lacks from NEMWeb's daily dispatch reports, after an exact match check."""
    import datetime as dt

    from gridcast.database.engine import get_engine
    from gridcast.ingestion.daily import DailyCheckFailed, ingest_daily
    from gridcast.ingestion.pipeline import parse_month_arg

    try:
        days = [dt.date.fromisoformat(d) for d in day]
    except ValueError as exc:
        raise typer.BadParameter(f"--day must be YYYY-MM-DD: {exc}") from exc
    cfg = load_config()
    try:
        result = ingest_daily(
            get_engine(), cfg, days, parse_month_arg(check_month), _regions(regions)
        )
    except DailyCheckFailed as exc:
        _echo_match_check(exc.check)
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _echo_match_check(result.check)
    typer.echo(
        tabulate(
            [[d, n] for d, n in result.loaded.items()],
            headers=["day (NEM)", "rows loaded"],
            tablefmt="github",
        )
    )
    typer.echo(f"check written to {result.report_path}")


def _echo_match_check(check: MatchCheck) -> None:
    typer.echo(
        f"match check {check.month} ({check.days} days, {', '.join(check.regions)}; loaded from "
        f"{', '.join(check.loaded_sources)}): intervals only in the daily reports "
        f"{check.only_daily}, only loaded {check.only_loaded}; passed: {check.passed}"
    )
    rows = [
        [
            c.column,
            c.compared,
            c.differing,
            "" if c.max_abs_difference is None else f"{c.max_abs_difference:g}",
            "yes" if c.must_match else "",
        ]
        for c in check.columns
    ]
    typer.echo(
        tabulate(
            rows,
            headers=["column", "compared", "differing", "max |diff|", "must match"],
            tablefmt="github",
        )
    )


@app.command("ingest-aemo")
def ingest_aemo(
    start: StartOpt = None,
    end: EndOpt = None,
    regions: RegionsOpt = None,
    source: Annotated[str, typer.Option(help="predispatch, p5min or all")] = "all",
) -> None:
    """Download, validate and load AEMO pre-dispatch demand forecasts (M2 benchmark only)."""
    from gridcast.database.engine import get_engine
    from gridcast.ingestion.aemo_forecasts import FORECAST_TABLES, ingest_aemo_range
    from gridcast.ingestion.pipeline import parse_month_arg

    tables = [t for t in FORECAST_TABLES if source in ("all", t.source)]
    if not tables:
        raise typer.BadParameter(
            f"source must be all or one of {[t.source for t in FORECAST_TABLES]}"
        )
    cfg = load_config()
    first = parse_month_arg(start or cfg.aemo.ingest_start_month)
    last = parse_month_arg(end or cfg.aemo.ingest_end_month)
    results = ingest_aemo_range(get_engine(), cfg, first, last, _regions(regions), tables)
    rows = [
        [f"{r.period:%Y-%m}", name, r.status, r.rows_parsed, r.rows_loaded]
        for name, months in results.items()
        for r in months
    ]
    typer.echo(
        tabulate(rows, headers=["month", "source", "status", "parsed", "loaded"], tablefmt="github")
    )
    failed = [
        (name, r) for name, months in results.items() for r in months if r.status != "success"
    ]
    for name, r in failed:
        typer.echo(f"FAILED {r.period:%Y-%m} {name}: {r.error}", err=True)
    if failed:
        raise typer.Exit(code=1)


@app.command("ingest-aemo-poe")
def ingest_aemo_poe(start: StartOpt = None, end: EndOpt = None, regions: RegionsOpt = None) -> None:
    """Download, validate and load AEMO's PDPASA POE demand forecasts (M3 benchmark only)."""
    from gridcast.database.engine import get_engine
    from gridcast.ingestion.aemo_poe import ingest_poe_range
    from gridcast.ingestion.pipeline import parse_month_arg

    cfg = load_config()
    first = parse_month_arg(start or cfg.aemo.ingest_start_month)
    last = parse_month_arg(end or cfg.aemo.ingest_end_month)
    results = ingest_poe_range(get_engine(), cfg, first, last, _regions(regions))
    rows = [[f"{r.period:%Y-%m}", r.status, r.rows_parsed, r.rows_loaded] for r in results]
    typer.echo(tabulate(rows, headers=["month", "status", "parsed", "loaded"], tablefmt="github"))
    failed = [r for r in results if r.status != "success"]
    for r in failed:
        typer.echo(f"FAILED {r.period:%Y-%m}: {r.error}", err=True)
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


@app.command("evaluate-holdout")
def evaluate_holdout(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    check_only: Annotated[
        bool, typer.Option("--check-only", help="Run the refusal checks only; fit nothing")
    ] = False,
) -> None:
    """M2 holdout, scored once (S5): frozen recipe, exact reproduction check, then scoring."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.evaluation.holdout import (
        HoldoutRefused,
        ReproductionFailed,
        git_state,
        preflight,
        run_holdout,
        store_holdout,
    )

    cfg = load_config()
    engine = get_engine()
    git = git_state(cfg.backtest.frozen_ml_commit)
    try:
        refs = preflight(engine, cfg, region, git)
    except HoldoutRefused as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"checks pass: config {cfg.backtest.frozen_ml_config_hash}, commit {git.sha}, "
        f"frozen code unchanged since {cfg.backtest.frozen_ml_commit}, holdout not yet scored"
    )
    for key, res in refs.items():
        typer.echo(f"  run 2 '{key}' batch {res.batch_id}")
    if check_only:
        return
    t0 = time.perf_counter()
    try:
        run = run_holdout(
            engine, cfg, region, git, cfg.data_path(cfg.ml.model_dir), references=refs
        )
    except (HoldoutRefused, ReproductionFailed) as exc:
        typer.echo(f"STOPPED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    batches = store_holdout(engine, cfg, region, run, git)
    rows = [
        [
            r.evaluation_set,
            r.retrain,
            len(r.models),
            f"{r.issue_times:,}",
            r.fits,
            "exact" if r.passed else "FAILED",
        ]
        for r in run.reproductions
    ]
    typer.echo(
        tabulate(
            rows,
            headers=["set", "retrain", "models", "dev issue times", "fits", "reproduction"],
            tablefmt="github",
        )
    )
    for es in run.sets:
        typer.echo(
            f"stored {es.run_config['evaluation_set']}: batch {batches[es.key]}, "
            f"{es.run_config['n_origins']:,} issue times, "
            f"{es.run_config['scored_pairs']:,} common pairs"
        )
    typer.echo(
        f"took {(time.perf_counter() - t0) / 3600:.1f} h; scores are in the database, "
        "`gridcast report-holdout` writes the report"
    )


@app.command("backtest-ml")
def backtest_ml(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    until: Annotated[
        str | None,
        typer.Option(help="Smoke run: stop before this NEM date (YYYY-MM-DD); nothing is stored"),
    ] = None,
    plus: Annotated[bool, typer.Option(help="Also run C⁺/D⁺")] = True,
    delay: Annotated[bool, typer.Option(help="Also run the publication-delay check")] = True,
) -> None:
    """M2 walk-forward: LightGBM + ridge (A, C, D, C⁺, D⁺); baselines rescored on the same pairs."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.forecasting.ml_backtest import EvaluationSet, run_ml_backtest, store_sets
    from gridcast.forecasting.origins import nem_midnight_utc

    cfg = load_config()
    engine = get_engine()
    t0 = time.perf_counter()
    smoke = until is not None

    def show(es: EvaluationSet) -> None:
        m = es.metrics[(es.metrics["segment"] == "all") & (es.metrics["metric"] == "mae")]
        rows = []
        for model in es.out.model_names:
            v = m[m["model"] == model].set_index("horizon")["value"]
            rows.append([model, *(v.get(h) for h in (1, 12, 48, 144, 288, 0))])
        pairs = es.run_config["scored_pairs"]
        n_issue = len(es.out.origins)
        typer.echo(f"\n[{es.key}] dev MAE (MW) on {pairs:,} common pairs, {n_issue:,} issue times")
        headers = ["model", "h=1", "h=12", "h=48", "h=144", "h=288", "all"]
        typer.echo(tabulate(rows, headers=headers, floatfmt=".1f", tablefmt="github"))

    sets = run_ml_backtest(
        engine,
        cfg,
        region,
        until=nem_midnight_utc(until) if until is not None else None,
        plus=plus,
        delay=delay,
        model_root=None if smoke else cfg.data_path(cfg.ml.model_dir),
        on_set=show,
    )
    if smoke:
        typer.echo(f"\nsmoke run (--until {until}): nothing stored")
    else:
        batches = store_sets(engine, cfg, region, sets)
        for key, batch in batches.items():
            typer.echo(f"stored {key}: batch {batch}")
    typer.echo(f"{time.perf_counter() - t0:.0f}s")


@app.command("intervals-conformal")
def intervals_conformal(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    workers: Annotated[int, typer.Option(help="Point models run in parallel")] = 4,
) -> None:
    """M3 step 1: conformal intervals around run 2's development forecasts (C and baselines)."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.intervals.batch import run_intervals_conformal, run_name

    cfg = load_config()
    t0 = time.perf_counter()
    batch, metrics = run_intervals_conformal(get_engine(), cfg, region, workers=workers)
    levels = cfg.intervals.levels
    rows = []
    for model, m in metrics.items():
        pooled = m[m["horizon"] == 0].set_index("metric")["value"]
        rows.append(
            [
                run_name(model),
                *(100 * pooled[f"coverage_{lv}"] for lv in levels),
                *(pooled[f"interval_score_{lv}"] for lv in levels),
                pooled["wis"],
            ]
        )
    headers = [
        "run",
        *(f"cov{lv} %" for lv in levels),
        *(f"IS{lv}" for lv in levels),
        "WIS",
    ]
    typer.echo(f"\nstored interval batch {batch}: development issue times, pooled over leads")
    typer.echo(tabulate(rows, headers=headers, floatfmt=".1f", tablefmt="github"))
    typer.echo(f"{time.perf_counter() - t0:.0f}s")


@app.command("intervals-cqr")
def intervals_cqr(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    workers: Annotated[int, typer.Option(help="Levels run in parallel")] = 3,
) -> None:
    """M3 step 3: CQR on the stored quantile LightGBM C batch (development); clean tree only."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.intervals.cqr_batch import RUN_NAME, run_intervals_cqr
    from gridcast.provenance import git_sha

    cfg = load_config()
    sha = git_sha()
    if sha is None or sha.endswith("-dirty"):
        typer.echo("refused: the CQR batch must be built from a clean commit (git status)")
        raise typer.Exit(code=1)
    t0 = time.perf_counter()
    run = run_intervals_cqr(get_engine(), cfg, region, workers=workers, sha=sha)
    levels = cfg.intervals.levels
    pooled = run.metrics[run.metrics["horizon"] == 0].set_index("metric")["value"]
    row = [
        RUN_NAME,
        *(100 * pooled[f"coverage_{lv}"] for lv in levels),
        *(pooled[f"interval_score_{lv}"] for lv in levels),
        pooled["wis"],
        100 * pooled["non_nested"],
    ]
    headers = [
        "run",
        *(f"cov{lv} %" for lv in levels),
        *(f"IS{lv}" for lv in levels),
        "WIS",
        "non-nested %",
    ]
    config = run.run_config
    typer.echo(
        f"\nstored CQR batch {run.batch_id} (run {run.run_id}) from QR batch {config['qr_batch']}: "
        f"development from {config['first_issue']}, {config['scored_pairs']:,} pairs, pooled"
    )
    typer.echo(tabulate([row], headers=headers, floatfmt=".1f", tablefmt="github"))
    crossed = ", ".join(f"{lv}%: {n:,}" for lv, n in run.result.crossed.items())
    typer.echo(f"crossed intervals output at their midpoint: {crossed}")
    typer.echo(f"{time.perf_counter() - t0:.0f}s")


@app.command("intervals-halfhour")
def intervals_halfhour(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    workers: Annotated[int, typer.Option(help="Runs in parallel")] = 4,
) -> None:
    """M3 step 5: conformal and CQR intervals for 30-minute means (dev); clean tree only."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.intervals.halfhour import run_intervals_halfhour
    from gridcast.provenance import git_sha

    cfg = load_config()
    sha = git_sha()
    if sha is None or sha.endswith("-dirty"):
        typer.echo("refused: the half-hour batch must be built from a clean commit (git status)")
        raise typer.Exit(code=1)
    t0 = time.perf_counter()
    run = run_intervals_halfhour(get_engine(), cfg, region, workers=workers, sha=sha)
    levels = cfg.intervals.levels
    rows = []
    for name, m in run.metrics.items():
        pooled = m[m["horizon"] == 0].set_index("metric")["value"]
        rows.append(
            [
                f"{name} (run {run.run_ids[name]})",
                *(100 * pooled[f"coverage_{lv}"] for lv in levels),
                *(pooled[f"interval_score_{lv}"] for lv in levels),
                pooled["wis"],
            ]
        )
    headers = ["run", *(f"cov{lv} %" for lv in levels), *(f"IS{lv}" for lv in levels), "WIS"]
    config = run.run_config
    typer.echo(
        f"\nstored half-hour interval batch {run.batch_id}: 30-minute means, development from "
        f"{config['first_issue']}, {config['scored_pairs']:,} periods, pooled"
    )
    typer.echo(tabulate(rows, headers=headers, floatfmt=".1f", tablefmt="github"))
    crossed = "; ".join(
        f"{name}: " + ", ".join(f"{lv}%: {n:,}" for lv, n in res.crossed.items())
        for name, res in run.results.items()
    )
    typer.echo(f"crossed intervals output at their midpoint: {crossed}")
    typer.echo(f"{time.perf_counter() - t0:.0f}s")


@app.command("intervals-freeze-check")
def intervals_freeze_check(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    reproduce: Annotated[
        bool, typer.Option(help="Recompute the stored dev conformal and CQR bounds (~8 min)")
    ] = True,
    workers: Annotated[int, typer.Option(help="Runs in parallel")] = 4,
) -> None:
    """M3 step 5: the interval recipe against its freeze record; exit 1 on any difference."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.intervals.freeze import freeze_state, reproduce_development

    cfg = load_config()
    t0 = time.perf_counter()
    state = freeze_state(cfg)
    typer.echo(
        f"interval config hash {state.config_hash} (frozen {state.frozen_config_hash}); "
        f"QR config hash {state.qr_config_hash} (frozen {state.frozen_qr_config_hash}); "
        f"HEAD {state.sha}, frozen commit {state.frozen_commit}"
    )
    failed = False
    for problem in state.problems:
        typer.echo(f"problem: {problem}")
        failed = True
    if reproduce:
        for r in reproduce_development(get_engine(), cfg, region, workers=workers):
            verdict = "reproduced exactly" if r.passed else "DIFFERS"
            detail = "" if r.passed else f" (bounds: {', '.join(r.differing_bounds) or 'equal'}; "
            if not r.passed:
                detail += f"selections {'equal' if r.selections_equal else 'differ'})"
            typer.echo(
                f"{r.run} (run {r.run_id}, batch {r.batch_id}, built at {r.git_sha}): "
                f"{r.issue_times:,} issue times, {verdict}{detail}"
            )
            failed |= not r.passed
    typer.echo(f"{time.perf_counter() - t0:.0f}s")
    if failed:
        raise typer.Exit(code=1)


@app.command("backtest-qr")
def backtest_qr(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    until: Annotated[
        str | None,
        typer.Option(help="Smoke run: stop before this NEM date (YYYY-MM-DD); nothing is stored"),
    ] = None,
) -> None:
    """M3 step 2: quantile LightGBM C walk-forward (development), reusing run 2's settings."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.forecasting.origins import nem_midnight_utc
    from gridcast.intervals.qr_batch import (
        RUN_NAME,
        cost_estimate,
        run_backtest_qr,
        score_development,
        smoke_metrics,
        store_qr,
    )
    from gridcast.intervals.quantile import quantile_key
    from gridcast.provenance import git_sha

    cfg = load_config()
    smoke = until is not None
    sha = git_sha()
    if not smoke and (sha is None or sha.endswith("-dirty")):
        typer.echo("refused: the full QR run must start from a clean commit (git status)")
        raise typer.Exit(code=1)
    engine = get_engine()
    t0 = time.perf_counter()
    run = run_backtest_qr(
        engine,
        cfg,
        region,
        until=nem_midnight_utc(until) if until is not None else None,
        model_root=None if smoke else cfg.data_path(cfg.ml.model_dir),
    )
    short = {"residual": "res", "direct": "dir", "anchored": "anc", "latest": "lat"}
    rows = []
    for rec in run.result.fits:
        fit_seconds = sum(q.score_seconds + q.refit_seconds for q in rec.quantiles)
        rows.append(
            [
                rec.retrain,
                rec.band,
                rec.rows,
                *(
                    f"{short[q.chosen['target_form']]}/{q.chosen['num_leaves']} "
                    f"{q.n_trees} {q.es_pinball:.1f}"
                    for q in rec.quantiles
                ),
                100 * rec.crossing_share,
                rec.point_refit_seconds + fit_seconds,
            ]
        )
    keys = [quantile_key(t) for t in run.result.taus]
    headers = ["retrain", "band", "rows", *keys, "crossed %", "fit s"]
    typer.echo(
        f"\n{RUN_NAME}: per (retrain, band) and quantile, the chosen form/leaves, trees and "
        "early-stopping pinball loss (MW); point forecasts reproduced run 2 exactly"
    )
    typer.echo(tabulate(rows, headers=headers, floatfmt=".2f", tablefmt="github"))

    levels = cfg.intervals.levels
    if smoke:
        m = smoke_metrics(run, cfg).set_index("metric")["value"]
        label = "smoke run, every scorable pair (pre-scoring issue times)"
    else:
        metrics, summary = score_development(run, cfg)
        m = metrics[metrics["horizon"] == 0].set_index("metric")["value"]
        label = f"development from {summary['first_issue']}, {summary['scored_pairs']:,} pairs"
    typer.echo(f"\nraw QR, pooled over leads ({label})")
    table = [
        [
            *(100 * m[f"coverage_{lv}"] for lv in levels),
            *(m[f"interval_score_{lv}"] for lv in levels),
            m["wis"],
            100 * m["crossing"],
        ]
    ]
    headers = [*(f"cov{lv} %" for lv in levels), *(f"IS{lv}" for lv in levels), "WIS"]
    typer.echo(tabulate(table, headers=[*headers, "crossed %"], floatfmt=".1f", tablefmt="github"))

    if smoke:
        est = cost_estimate(run)
        typer.echo(
            f"\nsmoke run (--until {until}): nothing stored. {est['smoke_hours']:.2f} h, of "
            f"which fits {est['smoke_fit_hours']:.2f} h. Full development run "
            f"({est['retrains']:.0f} retrains), estimated: {est['full_hours_by_rows']:.1f} h "
            f"scaled by training rows, {est['full_hours_by_run2']:.1f} h scaled by run 2's "
            "own fit-time growth"
        )
    else:
        run_id = store_qr(engine, cfg, region, run, metrics, summary, sha)
        typer.echo(f"\nstored QR batch {run.batch_id} (run {run_id})")
    typer.echo(f"{time.perf_counter() - t0:.0f}s")


@app.command("backtest-qr-holdout")
def backtest_qr_holdout(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    check_only: Annotated[
        bool, typer.Option("--check-only", help="Run the refusal checks only; fit nothing")
    ] = False,
) -> None:
    """M3 step 6a: quantile LightGBM C over the holdout, stored UNSCORED (no metric computed)."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.evaluation.holdout import HoldoutRefused
    from gridcast.intervals.freeze import freeze_state
    from gridcast.intervals.qr_holdout import (
        QrHoldoutRefused,
        QrHoldoutReproductionFailed,
        fit_headers,
        fit_rows,
        preflight,
        run_qr_holdout,
        store_qr_holdout,
    )
    from gridcast.intervals.quantile import ReproductionError

    cfg = load_config()
    engine = get_engine()
    state = freeze_state(cfg)
    try:
        inputs = preflight(engine, cfg, region, state)
    except QrHoldoutRefused as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"checks pass: QR recipe {state.qr_config_hash}, interval recipe {state.config_hash}, "
        f"commit {state.sha}, frozen code unchanged since {state.frozen_commit}, no QR holdout "
        f"batch yet. Development QR batch {inputs.dev_qr.batch_id}; holdout_main batch "
        f"{inputs.holdout_point.batch_id}; retrains "
        + ", ".join(r.strftime("%Y-%m-%d") for r in inputs.window.retrains)
    )
    if check_only:
        return
    t0 = time.perf_counter()
    try:
        run = run_qr_holdout(engine, cfg, region, inputs, cfg.data_path(cfg.ml.model_dir))
    except (HoldoutRefused, QrHoldoutRefused, QrHoldoutReproductionFailed, ReproductionError) as e:
        typer.echo(f"STOPPED: {e}", err=True)
        raise typer.Exit(code=1) from e
    run_id = store_qr_holdout(engine, cfg, region, run, state.sha)
    typer.echo(
        "\nper (retrain, band) and quantile: the chosen form/leaves, trees and early-stopping "
        "pinball loss (MW, on the training slice before the retrain)"
    )
    typer.echo(
        tabulate(
            fit_rows(run.result.fits),
            headers=fit_headers(run.result.taus),
            floatfmt=".2f",
            tablefmt="github",
        )
    )
    rep = run.reproduction
    typer.echo(
        f"\nreproduction at {rep.retrain}: {rep.fits} fit records and {rep.pairs:,} quantile "
        f"values of {rep.issue_times:,} development issue times equal step 2's batch "
        f"{rep.reference_batch}; point forecasts equal run 2's and holdout_main's at every "
        f"retrain ({run.run_config['point_reproduction']['pairs']:,} pairs)"
    )
    typer.echo(
        f"stored QR holdout batch {run.batch_id} (run {run_id}): "
        f"{run.run_config['n_origins']:,} issue times, of which "
        f"{run.run_config['holdout_issue_times']:,} in the holdout. Nothing was scored; "
        "`gridcast evaluate-holdout-intervals` scores it under the U5 lock"
    )
    typer.echo(f"took {(time.perf_counter() - t0) / 3600:.2f} h")


@app.command("weather-ablation")
def weather_ablation(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Weather-feature drop/add test (m2_plan.md §2.5) on the stored ML run; ~30-40 min."""
    import time

    import pandas as pd

    from gridcast.database.engine import get_engine
    from gridcast.forecasting.ablation import (
        fits_frame,
        flag_table,
        reproduction_ok,
        run_weather_ablation,
        write_ablation,
    )

    cfg = load_config()
    t0 = time.perf_counter()
    result = run_weather_ablation(get_engine(), cfg, region)
    path = write_ablation(cfg, result)
    flags = flag_table(fits_frame(result), result["flag_share"])
    with pd.option_context("display.width", 160):
        typer.echo(flags.round(4).to_string(index=False))
    typer.echo(f"stored early-stopping MAE reproduced: {reproduction_ok(result)}")
    typer.echo(f"wrote {path} ({time.perf_counter() - t0:.0f}s)")


@app.command()
def report(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write reports/m1_baseline.md and figures from the latest stored backtests."""
    from gridcast.database.engine import get_engine
    from gridcast.evaluation.report import write_report

    path = write_report(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("report-m2")
def report_m2(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write reports/m2_demand.md and figures from the stored ML run (runs the leakage tests)."""
    from gridcast.database.engine import get_engine
    from gridcast.evaluation.m2_report import write_report as write_m2

    path = write_m2(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("report-aemo")
def report_aemo(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Generate reports/m2_aemo.md: the frozen model vs AEMO's forecasts (development part)."""
    from gridcast.database.engine import get_engine
    from gridcast.evaluation.aemo_benchmark import write_report

    path = write_report(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("report-holdout")
def report_holdout(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Generate reports/m2_holdout.md from the stored holdout run (S5, replication, AEMO)."""
    from gridcast.database.engine import get_engine
    from gridcast.evaluation.holdout_report import write_report

    path = write_report(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("report-m3")
def report_m3(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write reports/m3_intervals.md and figures from the stored interval batches (U1-U4)."""
    from gridcast.database.engine import get_engine
    from gridcast.intervals.report import write_report as write_m3

    path = write_m3(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("report-m3-aemo")
def report_m3_aemo(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write reports/m3_aemo.md: our half-hour intervals against AEMO's POE band (dev part)."""
    from gridcast.database.engine import get_engine
    from gridcast.intervals.aemo_report import write_report as write_m3_aemo

    path = write_m3_aemo(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("feature-catalogue")
def feature_catalogue() -> None:
    """Write docs/feature_catalogue.md from the feature catalogue in code (no database)."""
    from gridcast.config import PROJECT_ROOT
    from gridcast.features.catalogue import render_catalogue

    path = PROJECT_ROOT / "docs" / "feature_catalogue.md"
    path.write_text(render_catalogue(load_config()), encoding="utf-8")
    typer.echo(f"wrote {path}")


@app.command("feature-audit")
def feature_audit(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write reports/m2_feature_audit.md: feature coverage measured on the stored data."""
    from gridcast.database.engine import get_engine
    from gridcast.features.audit import write_audit

    path = write_audit(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
