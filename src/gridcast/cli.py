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


@app.command("ingest-generation")
def ingest_generation(
    start: StartOpt = None,
    end: EndOpt = None,
    regions: RegionsOpt = None,
    day: Annotated[
        list[str] | None,
        typer.Option("--day", help="NEM-time day YYYY-MM-DD from the daily reports (repeatable)"),
    ] = None,
    check_month: Annotated[
        str | None,
        typer.Option("--check-month", help="Month YYYY-MM compared with the daily reports"),
    ] = None,
) -> None:
    """Load regional supply totals (wind, solar, dispatchable, battery) from the cached
    DISPATCHREGIONSUM months, then any --day from the daily reports after an exact match with
    MMSDM over --check-month (M4 step 1b; evidence only, never a model input)."""
    import datetime as dt

    from gridcast.database.engine import get_engine
    from gridcast.ingestion.generation import (
        GenerationCheckFailed,
        ingest_generation_daily,
        ingest_generation_range,
    )
    from gridcast.ingestion.pipeline import parse_month_arg

    if day and not check_month:
        raise typer.BadParameter("--day needs --check-month")
    try:
        days = [dt.date.fromisoformat(d) for d in day or []]
    except ValueError as exc:
        raise typer.BadParameter(f"--day must be YYYY-MM-DD: {exc}") from exc
    cfg = load_config()
    engine = get_engine()
    region_list = _regions(regions)
    first = parse_month_arg(start or cfg.ingestion.start_month)
    last = parse_month_arg(end or cfg.ingestion.end_month)
    results = ingest_generation_range(engine, cfg, first, last, region_list)
    rows = [
        [f"{r.period:%Y-%m}", r.status, r.rows_parsed, r.duplicates_dropped, r.rows_loaded]
        for r in results
    ]
    typer.echo(
        tabulate(
            rows, headers=["month", "status", "parsed", "duplicates", "loaded"], tablefmt="github"
        )
    )
    failed = [r for r in results if r.status != "success"]
    for r in failed:
        typer.echo(f"FAILED {r.period:%Y-%m}: {r.error}", err=True)
    if failed:
        raise typer.Exit(code=1)
    if not days:
        return
    assert check_month is not None
    try:
        daily = ingest_generation_daily(
            engine, cfg, days, parse_month_arg(check_month), region_list
        )
    except GenerationCheckFailed as exc:
        typer.echo(f"FAILED: {exc}\n{exc.check.as_dict()}", err=True)
        raise typer.Exit(code=1) from exc
    c = daily.check
    typer.echo(
        f"match check {c.month} ({c.days} days; loaded from {', '.join(c.loaded_sources)}): "
        f"{c.compared} rows compared, only daily {c.only_daily}, only loaded {c.only_loaded}, "
        f"differing {c.differing}; passed: {c.passed}"
    )
    typer.echo(
        tabulate(
            [[d, n] for d, n in daily.loaded.items()],
            headers=["day (NEM)", "rows loaded"],
            tablefmt="github",
        )
    )
    typer.echo(f"check written to {daily.report_path}")


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
    from gridcast.timeutils import NEM_TZ

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
        + ", ".join(r.tz_convert(NEM_TZ).strftime("%Y-%m-%d") for r in inputs.window.retrains)
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


@app.command("evaluate-holdout-intervals")
def evaluate_holdout_intervals(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    check_only: Annotated[
        bool, typer.Option("--check-only", help="Run the refusal checks only; build nothing")
    ] = False,
    workers: Annotated[int, typer.Option(help="Runs in parallel")] = 3,
) -> None:
    """M3 step 6b: the holdout intervals, scored ONCE (U5): reproduce development, then score."""
    import time

    from gridcast.database.engine import get_engine
    from gridcast.intervals.freeze import freeze_state
    from gridcast.intervals.holdout import (
        HoldoutIntervalsRefused,
        HoldoutIntervalsReproductionFailed,
        preflight,
        run_holdout_intervals,
        store_holdout_intervals,
    )

    cfg = load_config()
    engine = get_engine()
    state = freeze_state(cfg)
    try:
        inputs = preflight(engine, cfg, region, state)
    except HoldoutIntervalsRefused as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"checks pass: interval recipe {state.config_hash}, QR recipe {state.qr_config_hash}, "
        f"commit {state.sha}, frozen code unchanged since {state.frozen_commit}, holdout "
        "intervals not yet scored"
    )
    typer.echo(
        f"  run 2 {cfg.intervals.source_batch}; holdout_main {inputs.holdout_point.batch_id}; "
        f"QR {inputs.dev_qr.batch_id} + holdout QR {inputs.qr_holdout.batch_id}; development "
        f"interval batches {sorted({r.batch_id for r in inputs.dev_runs.values()})} and "
        f"{sorted({r.batch_id for r in inputs.dev_halfhour.values()})}"
    )
    if check_only:
        return
    t0 = time.perf_counter()
    try:
        run = run_holdout_intervals(engine, cfg, region, inputs, workers=workers)
    except (HoldoutIntervalsRefused, HoldoutIntervalsReproductionFailed) as exc:
        typer.echo(f"STOPPED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    batch, run_ids = store_holdout_intervals(engine, cfg, region, run, state.sha)
    rows = [
        [r.run, r.batch_id, f"{r.issue_times:,}", r.selections, "exact" if r.passed else "DIFFERS"]
        for r in run.reproductions
    ]
    typer.echo("\ndevelopment part of the continuous run against the stored development batches")
    typer.echo(
        tabulate(
            rows,
            headers=["run", "development batch", "issue times", "selections", "reproduction"],
            tablefmt="github",
        )
    )
    typer.echo(
        f"step 6a's quantiles equal step 2's on {run.qr_overlap_issue_times:,} development "
        "issue times"
    )
    typer.echo(
        f"stored holdout interval batch {batch}: {len(run_ids)} runs, {len(run.times):,} "
        f"holdout issue times ({run.times[0]} → {run.times[-1]}). Scores are in the database; "
        "`gridcast report-m3-holdout` writes the report"
    )
    typer.echo(f"took {(time.perf_counter() - t0) / 60:.1f} min")


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


@app.command("report-m3-holdout")
def report_m3_holdout(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write reports/m3_holdout.md from the stored holdout interval run (U5, replication, AEMO)."""
    from gridcast.database.engine import get_engine
    from gridcast.intervals.holdout_report import write_report as write_m3_holdout

    path = write_m3_holdout(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("anomaly-reference-events")
def anomaly_reference_events(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
) -> None:
    """Write docs/m4_reference_events.md: AEMO-flagged reference events for A2 (M4 step 1a)."""
    from gridcast.anomalies.reference import CLASS_P, DOC_PATH, write_reference_events
    from gridcast.database.engine import get_engine

    events = write_reference_events(get_engine(), load_config(), region)
    judged = [e.event_id for e in events if e.judged_by_a2]
    n_p = sum(e.cls == CLASS_P for e in events)
    typer.echo(
        f"{len(events)} events ({n_p} class P); A2 judges {len(judged)}: " + ", ".join(judged)
    )
    typer.echo(f"wrote {DOC_PATH}")


@app.command("anomalies-smoke")
def anomalies_smoke(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    weeks: Annotated[int, typer.Option(help="Pseudo-scored weeks at each warm-up's end")] = 8,
    starts: Annotated[int, typer.Option(help="Injection starts per detector")] = 200,
    version: Annotated[int, typer.Option(help="Anomaly recipe version (1 or 2)")] = 1,
) -> None:
    """M4 step 2 smoke run on the warm-up periods only (bugs, never tuning; no writes)."""
    from gridcast.anomalies.smoke import market_pass, residual_pass, summarise
    from gridcast.database.engine import get_engine

    cfg = load_config()
    engine = get_engine()
    for run_pass in (market_pass, residual_pass):
        sp = run_pass(engine, cfg, region, weeks, starts, version=version)
        for line in summarise(sp, cfg):
            typer.echo(line)


@app.command("evidence-smoke")
def evidence_smoke(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    weeks: Annotated[int, typer.Option(help="Pseudo-scored weeks at each warm-up's end")] = 8,
    version: Annotated[int, typer.Option(help="Anomaly recipe version (1 or 2)")] = 1,
) -> None:
    """M4 step 3b: evidence and summaries of the warm-up smoke events (completeness,
    banned phrases, E1, 5 example summaries; no writes)."""
    from gridcast.anomalies.smoke import evidence_lines, market_pass, residual_pass
    from gridcast.database.engine import get_engine

    cfg = load_config()
    engine = get_engine()
    for run_pass in (market_pass, residual_pass):
        sp = run_pass(engine, cfg, region, weeks, 0, inject=False, version=version)
        for line in evidence_lines(engine, cfg, region, sp):
            typer.echo(line)


@app.command("explain-smoke")
def explain_smoke(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    retrain: Annotated[
        str, typer.Option(help="Run 2 retrain (its issue times must lie in D2/D3's warm-up)")
    ] = "2022-11-01",
) -> None:
    """M4 step 3 SHAP smoke run on one retrain: E1 results and timings only (no writes)."""
    from gridcast.database.engine import get_engine
    from gridcast.explainability.smoke import run_explain_smoke, summarise

    cfg = load_config()
    for line in summarise(run_explain_smoke(get_engine(), cfg, region, retrain)):
        typer.echo(line)


@app.command("anomalies-freeze-check")
def anomalies_freeze_check(
    version: Annotated[int, typer.Option(help="Anomaly recipe version (1 or 2)")] = 1,
) -> None:
    """M4 step 4: the anomaly recipe against its freeze record; exit 1 on any difference."""
    from gridcast.anomalies.freeze import freeze_state

    state = freeze_state(load_config(), version=version)
    typer.echo(
        f"recipe v{version}: anomaly config hash {state.config_hash} "
        f"(frozen {state.frozen_config_hash}); "
        f"HEAD {state.sha}{' (dirty)' if state.dirty else ''}, "
        f"frozen commit {state.frozen_commit}"
    )
    for problem in state.problems:
        typer.echo(f"problem: {problem}")
    if state.problems:
        raise typer.Exit(1)
    typer.echo("the freeze holds")


@app.command("anomalies-dev")
def anomalies_dev(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    version: Annotated[int, typer.Option(help="Anomaly recipe version (1 or 2)")] = 1,
) -> None:
    """M4 step 4: the one development run of the frozen anomaly recipe (scores, injections,
    real-data cuts, evidence; stored). Refuses unless the freeze holds and the tree is clean."""
    from gridcast.anomalies.dev import DevRefused, run_dev, summary_lines
    from gridcast.database.engine import get_engine

    try:
        run = run_dev(get_engine(), load_config(), region, version)
    except DevRefused as exc:
        typer.echo(f"refused: {exc}")
        raise typer.Exit(1) from exc
    for line in summary_lines(run):
        typer.echo(line)


@app.command("explain-dev")
def explain_dev(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    replace: Annotated[
        bool, typer.Option(help="Replace a stored sample (the run is deterministic)")
    ] = False,
) -> None:
    """M4 step 5: the global SHAP sample of run 2's LightGBM C fits, with E1 on every pair
    (stored under <data root>/explain/dev/; nothing is stored if E1 fails)."""
    from gridcast.database.engine import get_engine
    from gridcast.explainability.global_run import GlobalRefused, run_global, summary_lines
    from gridcast.explainability.treeshap import GateError

    try:
        run = run_global(get_engine(), load_config(), region, "dev", replace)
    except GlobalRefused as exc:
        typer.echo(f"refused: {exc}")
        raise typer.Exit(1) from exc
    except GateError as exc:
        typer.echo(f"E1 failed, nothing stored: {exc}")
        raise typer.Exit(1) from exc
    for line in summary_lines(run):
        typer.echo(line)


@app.command("explain-holdout")
def explain_holdout(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    replace: Annotated[
        bool, typer.Option(help="Replace a stored sample (the run is deterministic)")
    ] = False,
) -> None:
    """M4 step 6: the global SHAP sample of holdout_main's LightGBM C fits on holdout issue
    times, with E1 on every pair (stored under <data root>/explain/holdout/; nothing is
    stored if E1 fails). Prints only the gates and timings."""
    from gridcast.database.engine import get_engine
    from gridcast.explainability.global_run import GlobalRefused, run_global, summary_lines
    from gridcast.explainability.treeshap import GateError

    try:
        run = run_global(get_engine(), load_config(), region, "holdout", replace)
    except GlobalRefused as exc:
        typer.echo(f"refused: {exc}")
        raise typer.Exit(1) from exc
    except GateError as exc:
        typer.echo(f"E1 failed, nothing stored: {exc}")
        raise typer.Exit(1) from exc
    for line in summary_lines(run):
        typer.echo(line)


@app.command("report-m4-explain")
def report_m4_explain(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
) -> None:
    """M4 step 5: reports/m4_explainability.md from the stored global SHAP sample and the
    development price associations (reported only; E1 on the global sample)."""
    from gridcast.database.engine import get_engine
    from gridcast.explainability.report import write_report as write_explain

    path = write_explain(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("report-m4-explain-holdout")
def report_m4_explain_holdout(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
) -> None:
    """M4 step 6: reports/m4_explainability_holdout.md from the stored holdout SHAP sample and
    the holdout price associations on the development bins (development beside)."""
    from gridcast.database.engine import get_engine
    from gridcast.explainability.holdout_report import write_report as write_explain_holdout

    path = write_explain_holdout(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("report-m4")
def report_m4(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    version: Annotated[int, typer.Option(help="Anomaly recipe version (1 or 2)")] = 1,
) -> None:
    """Write reports/m4_anomalies.md (v2: m4_anomalies_v2.md, post hoc, beside v1) from the
    stored development run (A1-A5, E1 at peaks)."""
    from gridcast.anomalies.report import write_report as write_m4
    from gridcast.database.engine import get_engine

    path = write_m4(get_engine(), load_config(), region, version=version)
    typer.echo(f"wrote {path}")


@app.command("report-m4-holdout")
def report_m4_holdout(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
) -> None:
    """M4 step 6: reports/m4_holdout.md from both stored holdout runs (v1 and v2 side by side,
    development beside; runs the causality suite for each version)."""
    from gridcast.anomalies.holdout_report import write_report as write_holdout
    from gridcast.database.engine import get_engine

    path = write_holdout(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("anomalies-reproduce")
def anomalies_reproduce(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    version: Annotated[int, typer.Option(help="Anomaly recipe version (1 or 2)")] = 1,
) -> None:
    """Recompute the stored development run with the current code and compare everything it
    stored (nothing is written); exit 1 on any difference."""
    from gridcast.anomalies.reproduce import reproduce, summary_lines
    from gridcast.database.engine import get_engine

    rep = reproduce(get_engine(), load_config(), region, version)
    for line in summary_lines(rep):
        typer.echo(line)
    if not rep.ok:
        raise typer.Exit(1)


@app.command("evaluate-holdout-anomalies")
def evaluate_holdout_anomalies(
    region: Annotated[str, typer.Option(help="Region id")] = "VIC1",
    version: Annotated[int, typer.Option(help="Anomaly recipe version (1 or 2)")] = 1,
    check_only: Annotated[
        bool,
        typer.Option("--check-only", help="Run the refusals and input checks only; score nothing"),
    ] = False,
) -> None:
    """M4 step 6: the holdout anomaly run, scored ONCE per recipe hash (A6). It reproduces the
    version's development run, checks the continuous run's development part, then scores and
    stores the holdout; `gridcast report-m4-holdout` computes the criteria."""
    import time

    from gridcast.anomalies.holdout import (
        HoldoutRefused,
        HoldoutReproductionFailed,
        check_lines,
        preflight,
        run_holdout,
        summary_lines,
    )
    from gridcast.database.engine import get_engine
    from gridcast.explainability.treeshap import GateError

    cfg = load_config()
    engine = get_engine()
    t0 = time.perf_counter()
    try:
        pre = preflight(engine, cfg, region, version)
        if check_only:
            for line in check_lines(engine, cfg, region, pre):
                typer.echo(line)
            typer.echo("checks pass; nothing was scored")
            return
        stored = run_holdout(engine, cfg, region, version)
    except HoldoutRefused as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except HoldoutReproductionFailed as exc:
        typer.echo(f"STOPPED, nothing stored: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except GateError as exc:
        typer.echo(f"E1 failed at a holdout demand peak, nothing stored: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in summary_lines(stored):
        typer.echo(line)
    typer.echo(f"took {(time.perf_counter() - t0) / 60:.1f} min")


@app.command("anomalies-diagnose-v1")
def anomalies_diagnose_v1(region: Annotated[str, typer.Option(help="Region id")] = "VIC1") -> None:
    """Write docs/m4_v1_diagnosis.md: why v1's D1, D4 and D6 failed (reported only; recomputes
    v1's own state from the stored development run, scores no alternative)."""
    from gridcast.anomalies.diagnosis import write_doc
    from gridcast.database.engine import get_engine

    path = write_doc(get_engine(), load_config(), region)
    typer.echo(f"wrote {path}")


@app.command("publish")
def publish(
    check_only: Annotated[
        bool,
        typer.Option("--check-only", help="Check what is stored against the sources; no writes"),
    ] = False,
    product: Annotated[
        list[str] | None,
        typer.Option(help="Only these products (repeatable; default: all four)"),
    ] = None,
    split: Annotated[
        list[str] | None, typer.Option(help="Only these splits: dev, holdout (repeatable)")
    ] = None,
) -> None:
    """M5: load the scored forecasts, intervals and price baselines into Postgres, then read
    every row back and check it against its parquet source (P1, P4, interval metrics)."""
    import time

    from gridcast.config import SERVING_PRODUCTS, SERVING_SPLITS
    from gridcast.database.engine import get_engine
    from gridcast.provenance import git_sha
    from gridcast.serving.publish import (
        SUMMARY_HEADERS,
        PublishRefused,
        summary_rows,
    )
    from gridcast.serving.publish import (
        publish as run_publish,
    )

    products = tuple(product or SERVING_PRODUCTS)
    splits = tuple(split or SERVING_SPLITS)
    unknown = (set(products) - set(SERVING_PRODUCTS)) | (set(splits) - set(SERVING_SPLITS))
    if unknown:
        raise typer.BadParameter(f"unknown product or split: {sorted(unknown)}")
    t0 = time.perf_counter()
    try:
        results = run_publish(get_engine(), load_config(), check_only, git_sha(), products, splits)
    except PublishRefused as exc:
        typer.echo(f"refused, nothing written: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(tabulate(summary_rows(results), headers=SUMMARY_HEADERS, tablefmt="github"))
    for r in results:
        if r.after is not None and not r.after.complete:
            typer.echo(f"{r.label}: {r.after.describe()}")
    typer.echo(f"took {(time.perf_counter() - t0) / 60:.1f} min")
    good = {"complete"} if check_only else {"success"}
    if any(r.status not in good for r in results):
        raise typer.Exit(code=1)


@app.command("api-schema")
def api_schema() -> None:
    """Write docs/openapi.json, the API contract a unit test checks the app against."""
    from gridcast.api.app import create_app
    from gridcast.api.schema import write_schema

    typer.echo(f"wrote {write_schema(create_app())}")


@app.command("api-smoke")
def api_smoke(
    url: Annotated[
        str, typer.Option(help="The running API (make api-smoke passes GRIDCAST_API_PORT's)")
    ] = "http://127.0.0.1:8000",
) -> None:
    """M5 step 4: /health, then recompute and explain at serving.smoke's issue times, against a
    running API (informal; not P3 or P5). Exit 1 unless every check passes."""
    import httpx

    from gridcast.api.smoke import render, run_smoke, save
    from gridcast.provenance import git_sha

    cfg = load_config()
    with httpx.Client(base_url=url, timeout=120) as client:
        result = run_smoke(client, cfg.serving.smoke, git_sha(), url)
    typer.echo(render(result))
    try:
        typer.echo(f"saved {save(result, cfg)}")
    except FileNotFoundError as exc:  # the data root is on an unmounted drive
        typer.echo(f"not saved: {exc}")
    if not result.passed:
        raise typer.Exit(code=1)


@app.command("api-verify")
def api_verify(
    url: Annotated[
        str,
        typer.Option(help="The running API container (make api-verify passes GRIDCAST_API_PORT's)"),
    ] = "http://127.0.0.1:8000",
) -> None:
    """M5 step 5: P2, P3, P5 and P6 against the running API container (m5_plan.md §3), saved
    under <data root>/serving/. Refused (exit 2, nothing saved) unless the image is the
    checkout's HEAD and both are clean; exit 1 if a criterion fails."""
    import json

    import httpx

    from gridcast.api.catalogue import read_only_engine
    from gridcast.api.schema import SCHEMA_PATH
    from gridcast.api.verify import VerifyRefused, render, run_verify, save
    from gridcast.config import get_settings
    from gridcast.provenance import git_sha

    cfg = load_config()
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    engine = read_only_engine(get_settings().database_url)
    try:
        with httpx.Client(base_url=url, timeout=120) as client:
            result = run_verify(client, engine, cfg, git_sha(), url, committed)
    except VerifyRefused as exc:
        typer.echo(f"refused, nothing measured or saved: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        engine.dispose()
    path = save(result, cfg)
    typer.echo(render(result))
    typer.echo(f"saved {path}")
    if not result.passed:
        raise typer.Exit(code=1)


@app.command("report-m5")
def report_m5() -> None:
    """M5 step 5: write reports/m5_api.md from every stored verification run (api-verify) and
    the publications records (P1, P4)."""
    from gridcast.api.catalogue import read_only_engine
    from gridcast.api.report import write_report
    from gridcast.config import get_settings

    engine = read_only_engine(get_settings().database_url)
    try:
        typer.echo(f"wrote {write_report(engine, load_config())}")
    finally:
        engine.dispose()


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
