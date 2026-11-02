"""Generate ``reports/m1_baseline.md`` (+ figures) from results stored in PostgreSQL.

Nothing in the report is typed by hand: every number is read from ``model_runs``,
``forecast_metrics``, ``market_intervals`` and ``dq_results``, so re-running the
pipeline regenerates an honest report.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.figure import Figure
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.evaluation.segments import describe_segments

# Categorical slots from the validated palette. Colour follows the model, never its rank.
MODEL_COLORS = {
    "persistence": "#2a78d6",
    "daily_naive": "#eb6834",
    "weekly_naive": "#1baf7a",
    "median_profile": "#eda100",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "daily_naive": "Daily naive",
    "weekly_naive": "Weekly naive",
    "median_profile": "28-day median profile",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)
UNITS = {"demand": "MW", "price": "$/MWh"}


@dataclass(slots=True)
class TargetResults:
    target: str
    batch_id: str
    runs: pd.DataFrame
    metrics: pd.DataFrame  # model, split, segment, horizon, metric, value, n

    @property
    def models(self) -> list[str]:
        return [str(m) for m in self.runs["model_name"]]

    def value(self, model: str, split: str, segment: str, horizon: int, metric: str) -> float:
        m = self.metrics
        sel = m[
            (m["model"] == model)
            & (m["split"] == split)
            & (m["segment"] == segment)
            & (m["horizon"] == horizon)
            & (m["metric"] == metric)
        ]
        return float(sel["value"].iloc[0]) if len(sel) else float("nan")

    def n(self, split: str, segment: str, horizon: int) -> int:
        m = self.metrics
        sel = m[(m["split"] == split) & (m["segment"] == segment) & (m["horizon"] == horizon)]
        return int(sel["n"].iloc[0]) if len(sel) else 0

    def by_horizon(self, split: str, metric: str) -> pd.DataFrame:
        """``horizon x model`` table for the ``all`` segment (horizons 1..H)."""
        m = self.metrics
        sel = m[
            (m["split"] == split)
            & (m["segment"] == "all")
            & (m["metric"] == metric)
            & (m["horizon"] > 0)
        ]
        piv = sel.pivot_table(index="horizon", columns="model", values="value")
        return piv[[c for c in self.models if c in piv.columns]]


def load_latest(engine: Engine, region_id: str, target: str) -> TargetResults:
    with engine.connect() as conn:
        batch = conn.execute(
            text(
                """
                SELECT batch_id FROM model_runs
                WHERE region_id = :r AND target = :t AND status = 'success'
                ORDER BY created_at DESC LIMIT 1
                """
            ),
            {"r": region_id, "t": target},
        ).scalar()
        if batch is None:
            raise LookupError(f"no backtest for {region_id}/{target}; run `gridcast backtest`")
        runs = pd.read_sql(
            text("SELECT * FROM model_runs WHERE batch_id = :b ORDER BY run_id"),
            conn,
            params={"b": batch},
        )
        metrics = pd.read_sql(
            text(
                """
                SELECT r.model_name AS model, m.split, m.segment, m.horizon, m.metric,
                       m.value, m.n
                FROM forecast_metrics m JOIN model_runs r USING (run_id)
                WHERE r.batch_id = :b
                """
            ),
            conn,
            params={"b": batch},
        )
    return TargetResults(target, str(batch), runs, metrics)


def data_facts(engine: Engine, region_id: str) -> dict[str, Any]:
    with engine.connect() as conn:
        facts = (
            conn.execute(
                text(
                    """
                    SELECT count(*) AS n_rows,
                           min(interval_end) AS first_end,
                           max(interval_end) AS last_end,
                           min(total_demand_mw) AS min_demand,
                           max(total_demand_mw) AS max_demand,
                           percentile_cont(0.5) WITHIN GROUP (ORDER BY rrp_aud_mwh) AS median_rrp,
                           min(rrp_aud_mwh) AS min_rrp,
                           max(rrp_aud_mwh) AS max_rrp,
                           avg((rrp_aud_mwh < 0)::int) AS share_negative
                    FROM market_intervals WHERE region_id = :r
                    """
                ),
                {"r": region_id},
            )
            .mappings()
            .one()
        )
        dq = pd.read_sql(
            text(
                """
                SELECT check_name, coalesce(region_id, '-') AS region, status, n_violations,
                       details->>'description' AS description, run_at
                FROM dq_results
                WHERE check_run_id = (SELECT check_run_id FROM dq_results
                                      ORDER BY run_at DESC LIMIT 1)
                ORDER BY result_id
                """
            ),
            conn,
        )
    return {**dict(facts), "dq": dq}


# --------------------------------------------------------------------------- formatting


def horizon_label(h: int) -> str:
    """Lead time of horizon ``h`` (5-min steps) as e.g. ``35 min``, ``2 h``, ``1 h 40 min``."""
    minutes = h * 5
    hours, rest = divmod(minutes, 60)
    if hours == 0:
        return f"{rest} min"
    return f"{hours} h" if rest == 0 else f"{hours} h {rest} min"


def horizon_ranges(horizons: list[int]) -> str:
    """Compress sorted horizons into contiguous lead-time ranges."""
    runs: list[tuple[int, int]] = []
    for h in sorted(horizons):
        if runs and h == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], h)
        else:
            runs.append((h, h))
    return ", ".join(
        horizon_label(a) if a == b else f"{horizon_label(a)}–{horizon_label(b)}" for a, b in runs
    )


def money(v: float, digits: int = 0) -> str:
    sign = "−" if v < 0 else ""
    return f"{sign}${abs(v):,.{digits}f}"


def _fmt(v: float, digits: int = 1) -> str:
    return "–" if pd.isna(v) else f"{v:,.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _ts(value: Any) -> str:
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


# ----------------------------------------------------------------------------- figures


def plot_mae_by_horizon(res: TargetResults, path: Path, split: str = "dev") -> None:
    """Line chart of MAE vs lead time, one line per baseline, direct-labelled."""
    table = res.by_horizon(split, "mae")
    hours = table.index.to_numpy() * 5 / 60
    fig = Figure(figsize=(9, 4.8), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    for model in table.columns:
        ax.plot(hours, table[model], color=MODEL_COLORS.get(model, INK), linewidth=2,
                label=MODEL_LABELS.get(model, model), solid_capstyle="round")  # fmt: skip
    top = float(table.to_numpy().max()) * 1.08
    ax.set_ylim(0, top)
    ax.set_xlim(0, 24)

    # Direct labels at the right edge, nudged apart so they never collide.
    ends = sorted((float(table[m].iloc[-1]), m) for m in table.columns)
    gap = top * 0.05
    placed: list[float] = []
    for y, model in ends:
        y_label = max(y, placed[-1] + gap) if placed else y
        placed.append(y_label)
        ax.annotate(MODEL_LABELS.get(model, model), xy=(24, y), xytext=(24.5, y_label),
                    color=INK_2, fontsize=8.5, va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6})  # fmt: skip

    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Forecast lead time (hours after issue time)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"MAE ({UNITS[res.target]})", color=INK_2, fontsize=9)
    period = "development period" if split == "dev" else f"{split} period"
    ax.set_title(f"Victoria {res.target}: baseline MAE by lead time ({period})",
                 color=INK, fontsize=11, loc="left", pad=30)  # fmt: skip
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
                       fontsize=8.5, ncol=len(table.columns), handlelength=1.6)  # fmt: skip
    for t in legend.get_texts():
        t.set_color(INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.subplots_adjust(left=0.08, right=0.79, top=0.84, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


# ----------------------------------------------------------------------------- tables


def horizon_table(res: TargetResults, split: str, horizons: list[int], metric: str) -> str:
    """Models x selected horizons (+ pooled); the lowest error per column is bold."""
    cols = [*horizons, 0]
    headers = ["Model"] + [horizon_label(h) for h in horizons] + ["All horizons"]
    grid = {m: [res.value(m, split, "all", h, metric) for h in cols] for m in res.models}
    rows = []
    for model, values in grid.items():
        cells = []
        for j, v in enumerate(values):
            column = [grid[m][j] for m in res.models if pd.notna(grid[m][j])]
            is_best = metric != "bias" and bool(column) and v == min(column)
            cells.append(f"**{_fmt(v)}**" if is_best else _fmt(v))
        rows.append([MODEL_LABELS.get(model, model), *cells])
    return _md_table(headers, rows)


def segment_table(res: TargetResults, split: str, metric: str, cfg: AppConfig) -> str:
    desc = describe_segments(cfg.evaluation)
    headers = ["Segment", "Pairs"] + [MODEL_LABELS.get(m, m) for m in res.models]
    rows = []
    for segment, text_ in desc.items():
        values = [res.value(m, split, segment, 0, metric) for m in res.models]
        finite = [v for v in values if pd.notna(v)]
        best = min(finite) if finite else None
        cells = [f"**{_fmt(v)}**" if v == best else _fmt(v) for v in values]
        rows.append([text_, f"{res.n(split, segment, 0):,}", *cells])
    return _md_table(headers, rows)


def half_hour_position_table(res: TargetResults, split: str = "dev") -> str:
    """Mean per-horizon MAE grouped by the target's 5-minute slot within the half hour.

    Origins sit on :00/:30, so horizon h targets intervals ending (5h mod 30) minutes
    past the half hour. For origin-independent baselines this isolates how error varies
    with the target's position inside each 30-minute block.
    """
    table = res.by_horizon(split, "mae")
    minute = (table.index.to_numpy() * 5) % 30
    grouped = table.groupby(minute).mean()
    models = [m for m in ("daily_naive", "weekly_naive", "median_profile") if m in grouped]
    headers = ["Interval ends at (min past :00/:30)"] + [MODEL_LABELS[m] for m in models]
    rows = [[f":{int(k):02d}", *(_fmt(grouped.loc[k, m]) for m in models)] for k in grouped.index]
    return _md_table(headers, rows)


def best_per_horizon(res: TargetResults, split: str = "dev") -> pd.Series:
    """Model with the lowest MAE at each horizon (ties go to the first model listed)."""
    return res.by_horizon(split, "mae").idxmin(axis=1)


def crossover_horizon(res: TargetResults, a: str, b: str, split: str = "dev") -> int | None:
    """First horizon at which model ``b`` has a strictly lower MAE than model ``a``."""
    table = res.by_horizon(split, "mae")
    better = table.index[table[b] < table[a]]
    return int(better.min()) if len(better) else None


def summary_lines(res: TargetResults) -> list[str]:
    best = best_per_horizon(res)
    lines = []
    for model in res.models:
        hs = [int(h) for h in best.index[best == model]]
        if hs:
            lines.append(
                f"- **{MODEL_LABELS.get(model, model)}** has the lowest MAE at {len(hs)} of "
                f"{len(best)} lead times: {horizon_ranges(hs)}."
            )
    if {"persistence", "daily_naive"} <= set(res.models):
        h = crossover_horizon(res, "persistence", "daily_naive")
        if h:
            lines.append(
                f"- Daily naive first beats persistence at a lead time of {horizon_label(h)}."
            )
    return lines


# ----------------------------------------------------------------------------- report


def _provenance(facts: dict[str, Any], demand: TargetResults, price: TargetResults) -> str:
    run_cfg = demand.runs["config"].iloc[0]
    setup = run_cfg["setup"]
    shas = sorted({str(s) for s in [*demand.runs["git_sha"], *price.runs["git_sha"]]})
    rows = [
        ["Region", str(demand.runs["region_id"].iloc[0])],
        [
            "Data",
            "AEMO MMSDM `DISPATCHREGIONSUM` + `DISPATCHPRICE`, "
            f"{facts['n_rows']:,} five-minute intervals",
        ],
        ["Data window (interval end)", f"{_ts(facts['first_end'])} → {_ts(facts['last_end'])}"],
        [
            "Forecast origins",
            f"{run_cfg['n_origins']:,} (every 30 min), "
            f"{run_cfg['n_folds']} monthly walk-forward folds",
        ],
        [
            "Horizons",
            f"1–{setup['horizon_steps']} × 5 min (24 h); data latency "
            f"{setup['latency_steps']} intervals",
        ],
        [
            "Scored (origin, horizon) pairs per model",
            f"{run_cfg['scored_pairs']:,} of {run_cfg['total_pairs']:,}",
        ],
        [
            "Development period (origins)",
            f"{run_cfg['eval_start']} → {run_cfg['holdout_start']} (NEM time)",
        ],
        ["Holdout period (origins)", f"from {run_cfg['holdout_start']} (NEM time)"],
        ["Demand batch", f"`{demand.batch_id}`"],
        ["Price batch", f"`{price.batch_id}`"],
        ["Git commit(s)", ", ".join(f"`{s}`" for s in shas)],
    ]
    return _md_table(["Item", "Value"], rows, "ll")


def _data_quality(facts: dict[str, Any]) -> list[str]:
    dq: pd.DataFrame = facts["dq"]
    rows = [
        [
            f"{r['description']} (`{r['check_name']}`)",
            str(r["region"]),
            str(r["status"]).upper(),
            f"{int(r['n_violations']):,}",
        ]
        for r in dq.to_dict(orient="records")
    ]
    return [
        f"Latest data-quality run: {_ts(dq['run_at'].iloc[0])}. Fail-level checks must be "
        "zero. Warn-level checks report real market states, not data errors; those states "
        "are evaluated as separate segments.\n",
        _md_table(["Check", "Region", "Status", "Count"], rows, "llrr"),
        f"\nDemand ranged {facts['min_demand']:,.0f}–{facts['max_demand']:,.0f} MW. Price "
        f"ranged {money(facts['min_rrp'])} to {money(facts['max_rrp'])}/MWh with a median of "
        f"{money(facts['median_rrp'], 2)}/MWh, and "
        f"{100 * facts['share_negative']:.1f}% of intervals had a negative price.\n",
    ]


def build_report(engine: Engine, cfg: AppConfig, region_id: str) -> str:
    reports_dir = cfg.resolve(cfg.paths.reports_dir)
    facts = data_facts(engine, region_id)
    demand = load_latest(engine, region_id, "demand")
    price = load_latest(engine, region_id, "price")
    horizons = cfg.evaluation.horizons_reported
    for res in (demand, price):
        plot_mae_by_horizon(res, reports_dir / "figures" / f"m1_{res.target}_mae_by_horizon.png")

    out: list[str] = [
        "# Milestone 1: Victorian baseline forecasts\n",
        "> **Generated** by `gridcast report` on "
        f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M UTC} from results stored in PostgreSQL. "
        "Do not edit by hand: re-run the pipeline instead.\n",
        "## Provenance\n",
        _provenance(facts, demand, price),
        "",
        "## Data quality\n",
        *_data_quality(facts),
        "## How to read the results\n",
        "- Each origin issues a 24-hour forecast (288 five-minute steps). Errors are grouped "
        "by **lead time** (horizon) and pooled over every origin.\n"
        "- Every model is scored on the **same** (origin, horizon) pairs.\n"
        "- Error = forecast − actual, so positive bias means over-forecasting.\n"
        "- **Bold** marks the lowest error in each column. Taken per lead time, the best "
        "baseline forms the *baseline envelope* that any ML model must beat.\n"
        "- Price spikes, negative prices and interventions are defined by what actually "
        "happened (*ex-post*). They describe where models fail; they cannot be used to "
        "choose a model at forecast time.\n",
    ]

    for res, extra in ((demand, "mape"), (price, "medae")):
        unit = UNITS[res.target]
        title = res.target.capitalize()
        extra_label = "MAPE (%)" if extra == "mape" else f"Median absolute error ({unit})"
        out += [
            f"## {title} baselines\n",
            f"![{title} MAE by lead time](figures/m1_{res.target}_mae_by_horizon.png)\n",
            *summary_lines(res),
            "",
            f"### MAE ({unit}), development period\n",
            horizon_table(res, "dev", horizons, "mae"),
            f"\n### MAE ({unit}), holdout period\n",
            horizon_table(res, "holdout", horizons, "mae"),
            f"\n### RMSE ({unit}), development period\n",
            horizon_table(res, "dev", horizons, "rmse"),
            f"\n### {extra_label}, development period\n",
            horizon_table(res, "dev", horizons, extra),
            f"\n### Bias ({unit}, forecast − actual), development period\n",
            horizon_table(res, "dev", horizons, "bias"),
            f"\n### MAE ({unit}) by segment, all horizons pooled, development period\n",
            segment_table(res, "dev", "mae", cfg),
            "",
        ]
        if res.target == "price":
            out += [
                "### Price MAE by position within the half hour, development period\n",
                "Origins fall on :00 and :30, so each lead time targets a fixed 5-minute slot "
                "inside a half-hour block. These baselines don't depend on the origin, so the "
                "table isolates how their error changes with the target's position in the "
                "block (mean per-horizon MAE, $/MWh).\n",
                half_hour_position_table(res),
                "",
            ]

    out += [
        "## Notes and limitations\n",
        "- All of these are baselines. Nothing here is fitted to the holdout period, which "
        "stays reserved for the final evaluation of later ML models.\n"
        "- Seasonal naive uses fixed absolute lags (exactly 24 h / 7 d in UTC). On the "
        'two days a year when clocks change, "same time yesterday" is one local hour off. '
        "DST-change days are always Sundays in April or October, so the `dst_change_day` "
        "segment also mixes in day-type and seasonal effects. It cannot isolate the DST "
        "cost on its own.\n"
        "- `TOTALDEMAND` excludes rooftop PV, so midday demand partly reflects solar output "
        "(see `docs/data_sources.md`).\n"
        "- Backtests use final (revised) archive prices, which can differ slightly from "
        "what a live system saw in real time.\n",
    ]
    return "\n".join(out)


def write_report(engine: Engine, cfg: AppConfig, region_id: str) -> Path:
    path = cfg.resolve(cfg.paths.reports_dir) / "m1_baseline.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(engine, cfg, region_id))
    return path
