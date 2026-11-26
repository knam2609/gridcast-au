"""``gridcast report-m5``: ``reports/m5_api.md`` from the stored verification runs (plan §3.1).

* **P1 and P4** come from the latest ``publications`` record of each (product, split), which
  ``gridcast publish`` wrote after reading every row back.
* **P2, P3, P5 and P6** come from ``<data root>/serving/verify_*.json`` (``gridcast
  api-verify``). Every stored run is shown: the first run's verdicts lead, and each later run
  is listed with the commits between it and the run before (a rerun follows a change).
* **Seen before the first run:** the step-4 smoke checks, from ``<data root>/logs/
  api_smoke_*.json``. The question round's cost model of explain is in the plan.

Tables only. Every number is read from those files and records, none typed by hand.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, text

from gridcast.api.meta import DISCLOSURES
from gridcast.config import PROJECT_ROOT, SERVING_PRODUCTS, SERVING_SPLITS, AppConfig

REPORT = PROJECT_ROOT / "reports" / "m5_api.md"
CRITERIA = ("P1", "P2", "P3", "P4", "P5", "P6")
PASS_IF = {
    "P1": "every published value equals its parquet source exactly (every row read back); row "
    "counts = issue times × 288 − missing pairs; interval hit counts from Postgres equal M3's "
    "evaluation on the parquet",
    "P2": "on the seeded sample, every value served by the forecast, price, evolution and "
    "actuals endpoints equals its source, and every v2 event (paged and by id) equals its row",
    "P3": "for the same issue times, every lead recomputed equals the stored float32 (gaps "
    "included), and explain answers with E1's gate passed for every one",
    "P4": "lo95 ≤ lo80 ≤ lo50 ≤ hi50 ≤ hi80 ≤ hi95 in 100% of published interval pairs",
    "P5": "client-side p95 ≤ {stored} ms for each stored endpoint ({stored_n} requests each) and "
    "≤ {model} ms for recompute and explain ({model_n} each), after {warm} warm-ups each",
    "P6": "the served schema equals `docs/openapi.json`; every endpoint has a typed response; "
    "the invalid-input table returns its documented 4xx with a message; the search over "
    "query parameters finds no 500",
}
ENDPOINT_TITLES = {
    "forecast": "`/v1/forecasts/demand`",
    "evolution": "`/v1/forecasts/demand/evolution`",
    "price": "`/v1/forecasts/price`",
    "actuals": "`/v1/actuals` (one day)",
    "anomalies": "`/v1/anomalies` (one month, limit 100)",
    "event": "`/v1/anomalies/{id}`",
    "recompute": "`/v1/forecasts/demand/recompute`",
    "explain": "`/v1/forecasts/demand/explain` (all leads)",
}


def n(v: Any) -> str:
    return f"{int(v):,}"


def ms(v: float) -> str:
    return f"{v:,.1f}"


def verdict(ok: bool) -> str:
    return "**PASS**" if ok else "**FAIL**"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(":---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- inputs


def load_runs(directory: Path) -> list[dict[str, Any]]:
    """Every stored verification run, oldest first."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("verify_*.json")]
    return sorted(runs, key=lambda r: r["started_at"])


def load_smokes(directory: Path, before: str) -> list[dict[str, Any]]:
    """The smoke checks started before ``before`` (UTC ISO), oldest first."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in directory.glob("api_smoke_*.json")]
    return sorted((r for r in runs if r["started_at"] < before), key=lambda r: r["started_at"])


def load_publications(engine: Engine, region: str) -> list[dict[str, Any]]:
    """The latest publication of each (product, split), in the products' order."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT ON (product, split) publication_id, product, split, run_id,
                       source_run_id, first_issue, last_issue, issue_times, rows_expected,
                       missing_pairs, rows_inserted, git_sha, status, published_at, checks
                FROM publications WHERE region_id = :r
                ORDER BY product, split, publication_id DESC
                """
            ),
            {"r": region},
        ).all()
    by_key = {(r.product, r.split): dict(r._mapping) for r in rows}
    return [by_key[(p, s)] for p in SERVING_PRODUCTS for s in SERVING_SPLITS if (p, s) in by_key]


def git_log(a: str, b: str) -> list[str]:
    """``<short sha> <subject>`` of each commit in a..b (empty if git cannot say)."""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%h %s", f"{a}..{b}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in out.splitlines() if line.strip()]


# --------------------------------------------------------------------------- criteria


def p1_p4(pubs: Sequence[dict[str, Any]], n_leads: int) -> dict[str, tuple[bool, str]]:
    """P1 and P4 from the publications (each check a stored result of ``publish``)."""
    want = len(SERVING_PRODUCTS) * len(SERVING_SPLITS)
    rows_read = sum(int((p["checks"] or {}).get("p1", {}).get("rows_read", 0)) for p in pubs)
    bad = []
    counts_ok = 0
    for p in pubs:
        c = p["checks"] or {}
        p1 = c.get("p1") or {}
        label = f"{p['product']}/{p['split']}"
        if p["status"] != "success":
            bad.append(f"{label}: {p['status']}")
        if not (p1.get("complete") and p1.get("exact")):
            bad.append(f"{label}: {p1.get('value_mismatches')} values differ")
        expected = int(p["issue_times"]) * n_leads - int(p["missing_pairs"])
        if p1.get("rows_read") == p["rows_expected"] == expected:
            counts_ok += 1
        else:
            bad.append(f"{label}: rows {p1.get('rows_read')}, expected {expected}")
        if p["product"] == "demand_interval":
            m = c.get("interval_metrics") or {}
            if m.get("db_equals_parquet") is not True:
                bad.append(f"{label}: interval hit counts differ from the parquet's")
    intervals = [p for p in pubs if p["product"] == "demand_interval"]
    metric_splits = [
        p["split"]
        for p in intervals
        if ((p["checks"] or {}).get("interval_metrics") or {}).get("db_equals_parquet") is True
    ]
    p1_ok = len(pubs) == want and not bad
    p1_text = (
        f"{n(rows_read)} rows read back over {len(pubs)} of {want} publications, "
        + ("every value equal to its parquet" if p1_ok else "; ".join(bad))
        + f"; row counts = issue times × {n_leads} − missing pairs for {counts_ok} of "
        f"{len(pubs)}; interval hit counts from Postgres equal M3's on the parquet for "
        f"{len(metric_splits)} of {len(intervals)} splits"
    )
    pairs = sum(
        int((p["checks"] or {}).get("sql", {}).get("p4", {}).get("pairs", 0)) for p in intervals
    )
    non_nested = sum(
        int((p["checks"] or {}).get("sql", {}).get("p4", {}).get("non_nested", -1))
        for p in intervals
    )
    p4_ok = len(intervals) == len(SERVING_SPLITS) and non_nested == 0 and pairs > 0
    p4_text = (
        f"{n(non_nested)} of {n(pairs)} published interval pairs are not nested (SQL over every "
        "row; the table's CHECK also enforces it)"
    )
    return {"P1": (p1_ok, p1_text), "P4": (p4_ok, p4_text)}


def _checks(run: dict[str, Any], criterion: str) -> list[dict[str, Any]]:
    return [c for c in run["checks"] if c["criterion"] == criterion]


def _get(run: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in run["checks"] if c["name"] == name)


def _tail(c: dict[str, Any]) -> str:
    bad = []
    if c["mismatches"]:
        bad.append(f"{n(c['mismatches'])} mismatches")
    if c["bad_responses"]:
        bad.append(f"{n(c['bad_responses'])} unexpected responses")
    return ", ".join(bad) if bad else "0 mismatches"


def p2_text(run: dict[str, Any]) -> str:
    d, p = _get(run, "demand forecasts"), _get(run, "price baselines")
    e, a = _get(run, "forecast evolution"), _get(run, "actuals")
    an = _get(run, "anomaly events")
    priced = p["requests"] - p["extra"].get("not_published", 0)
    parts = [
        f"demand: {n(d['requests'])} issue times, {n(d['values_compared'])} values ({_tail(d)})",
        f"price: {n(priced)} issue times, {n(p['values_compared'])} values, and "
        f"{n(p['extra'].get('not_published', 0))} after the baselines' last issue time answered "
        f"404 as required ({_tail(p)})",
        f"evolution: {n(e['requests'])} targets, {n(e['extra'].get('items', 0))} items, "
        f"{n(e['values_compared'])} values ({_tail(e)})",
        f"actuals: {n(a['requests'])} days, {n(a['extra'].get('intervals', 0))} intervals, "
        f"{n(a['fields_compared'])} fields ({_tail(a)})",
        f"anomalies: {n(an['extra'].get('events', 0))} v2 events paged "
        f"({n(an['extra'].get('pages', 0))} pages of {an['extra'].get('limit')}) and each by id, "
        f"{n(an['fields_compared'])} fields ({_tail(an)})",
    ]
    return "; ".join(parts)


def p3_text(run: dict[str, Any], tolerance: float) -> str:
    r, e = _get(run, "recompute"), _get(run, "explain")
    x = e["extra"]
    return (
        f"recompute: {n(r['extra'].get('pairs_equal', 0))} of {n(r['extra'].get('pairs', 0))} "
        f"pairs equal ({_tail(r)}); explain: {n(x.get('explained', 0))} of "
        f"{n(x.get('issue_times', 0))} issue times explained with the gate passed, "
        f"{n(x.get('leads', 0))} leads, each re-added within {tolerance:g} ({_tail(e)})"
    )


def p5_text(run: dict[str, Any]) -> str:
    parts = []
    for name, e in run["p5"]["endpoints"].items():
        bad = f", {e['non_200']} not 200" if e["non_200"] else ""
        mark = "" if e["passed"] else " (FAIL)"
        parts.append(f"{name} {ms(e['p95_ms'])} ms{bad}{mark}")
    return "p95: " + "; ".join(parts)


def p6_text(run: dict[str, Any]) -> str:
    s, i = _get(run, "schema"), _get(run, "invalid inputs (422)")
    nf, h = _get(run, "nothing published (404)"), _get(run, "no-500 search")
    statuses = ", ".join(f"{k}: {v}" for k, v in h["extra"].get("statuses", {}).items())
    return (
        f"served schema {'equals' if s['passed'] else 'differs from'} `docs/openapi.json`, "
        f"{n(s['extra'].get('paths', 0))} paths, each GET with a typed 200 ({_tail(s)}); "
        f"{n(i['extra'].get('cases', 0))} invalid inputs → 422 with their message ({_tail(i)}); "
        f"{n(nf['extra'].get('cases', 0))} valid but unpublished → 404 ({_tail(nf)}); search: "
        f"{n(h['requests'])} examples ({statuses}), {_tail(h)}"
    )


def criteria_rows(
    run: dict[str, Any], pubs: Sequence[dict[str, Any]], cfg: AppConfig
) -> list[list[str]]:
    v = run["settings"]
    stored = p1_p4(pubs, cfg.forecast.horizon_steps)
    verdicts = run["verdicts"]
    texts = {
        "P2": p2_text(run),
        "P3": p3_text(run, v["sum_tolerance"]),
        "P5": p5_text(run),
        "P6": p6_text(run),
    }
    pass_if = dict(PASS_IF)
    pass_if["P5"] = pass_if["P5"].format(
        stored=f"{v['stored_p95_ms']:g}",
        model=f"{v['model_p95_ms']:g}",
        stored_n=v["stored_requests"],
        model_n=v["model_requests"],
        warm=v["warmup_requests"],
    )
    rows = []
    for c in CRITERIA:
        if c in stored:
            ok, body = stored[c]
        else:
            ok, body = bool(verdicts[c]), texts[c]
        rows.append([c, pass_if[c], f"{verdict(ok)}: {body}"])
    return rows


# --------------------------------------------------------------------------- sections


def _runs_section(runs: Sequence[dict[str, Any]], log: Callable[[str, str], list[str]]) -> str:
    rows = []
    prev: str | None = None
    for k, r in enumerate(runs, 1):
        sha = r["build"].get("git_sha") or "?"
        if prev is None:
            changes = "the first run"
        elif prev == sha:
            changes = "none (same commit)"
        else:
            commits = log(prev, sha)
            changes = (
                "<br>".join(f"`{c.split(' ', 1)[0]}` {c.split(' ', 1)[1]}" for c in commits)
                or f"`{prev[:7]}..{sha[:7]}`"
            )
        v = r["verdicts"]
        stable = "" if r.get("build_unchanged", True) else " (the build changed during the run)"
        rows.append(
            [
                k,
                r["started_at"],
                f"`{sha[:12]}`{stable}",
                changes,
                *("pass" if v[c] else "**fail**" for c in ("P2", "P3", "P5", "P6")),
            ]
        )
        prev = sha
    return table(
        [
            "Run",
            "Started (UTC)",
            "Image commit",
            "Changes since the run before",
            "P2",
            "P3",
            "P5",
            "P6",
        ],
        rows,
    )


def _latency_section(run: dict[str, Any]) -> str:
    rows = []
    for name, e in run["p5"]["endpoints"].items():
        rows.append(
            [
                ENDPOINT_TITLES.get(name, name),
                n(e["n"]),
                ms(e["p50_ms"]),
                ms(e["p95_ms"]),
                ms(e["p99_ms"]),
                ms(e["max_ms"]),
                f"{e['limit_ms']:g}",
                n(e["non_200"]),
                f"{e['median_kib']:,.1f}",
                verdict(e["passed"]),
            ]
        )
    return table(
        [
            "Endpoint",
            "Timed",
            "p50 (ms)",
            "p95 (ms)",
            "p99 (ms)",
            "max (ms)",
            "p95 limit (ms)",
            "Not 200",
            "Median KiB",
            "P5",
        ],
        rows,
    )


def _checks_section(run: dict[str, Any]) -> str:
    rows = []
    examples = []
    for c in run["checks"]:
        rows.append(
            [
                c["criterion"],
                c["name"],
                n(c["requests"]),
                n(c["values_compared"]),
                n(c["fields_compared"]),
                n(c["mismatches"]),
                n(c["bad_responses"]),
                f"{c['seconds']:,.0f}",
                verdict(c["passed"]),
            ]
        )
        if c["examples"]:
            examples.append(
                f"- **{c['criterion']} {c['name']}:** "
                + "; ".join(f"`{e}`" for e in c["examples"][:5])
            )
    out = table(
        [
            "",
            "Check",
            "Requests",
            "Values compared",
            "Fields compared",
            "Mismatches",
            "Unexpected responses",
            "Seconds",
            "Result",
        ],
        rows,
    )
    if examples:
        out += "\n\nFirst mismatches (up to 5 per check; the run stores up to 20):\n\n" + "\n".join(
            examples
        )
    return out


def _publications_section(pubs: Sequence[dict[str, Any]]) -> str:
    rows = []
    for p in pubs:
        c = p["checks"] or {}
        p1 = c.get("p1") or {}
        p4 = (c.get("sql") or {}).get("p4")
        rows.append(
            [
                f"{p['product']} / {p['split']}",
                p["run_id"],
                p["source_run_id"],
                f"{p['first_issue']:%Y-%m-%d %H:%M} → {p['last_issue']:%Y-%m-%d %H:%M}",
                n(p["issue_times"]),
                n(p["rows_expected"]),
                n(p["missing_pairs"]),
                f"{n(p1.get('equal', 0))} of {n(p1.get('rows_read', 0))}",
                "–" if p4 is None else f"{n(p4['non_nested'])} of {n(p4['pairs'])}",
                f"`{(p['git_sha'] or '?')[:7]}`",
            ]
        )
    return table(
        [
            "Product / split",
            "Run",
            "Source run",
            "Issue times (UTC)",
            "Issue times",
            "Rows",
            "Missing pairs",
            "P1: equal rows",
            "P4: not nested",
            "Publish commit",
        ],
        rows,
    )


def _samples_section(run: dict[str, Any]) -> str:
    s = run["samples"]
    rows = [
        [
            f["split"],
            f["retrain"],
            f"{f['start']} → {f['end']}",
            n(f["candidates"]),
            n(len(f["issue_times"])),
        ]
        for f in s["folds"]
    ]
    return (
        table(
            ["Split", "Retrain", "Issue times [start, end) (UTC)", "With a stored lead", "Drawn"],
            rows,
        )
        + f"\n\nEvolution: {n(len(s['evolution_targets']))} targets; actuals: "
        f"{n(len(s['actual_days']))} NEM days. Seed {run['settings']['seed']}."
    )


def _reported_section(run: dict[str, Any]) -> str:
    rep = run["reported"]
    rows = [
        [
            split,
            n(o["pairs"]),
            n(o["below"]),
            n(o["above"]),
            f"{100 * o['share']:.1f}%" if o["share"] is not None else "–",
        ]
        for split, o in rep["point_outside_50"].items()
    ]
    out = [
        "**The point outside the 50% interval** (SQL over every published pair; reported "
        "because CQR's bounds are not centred on the point, plan §3.1):",
        "",
        table(["Split", "Pairs", "Point below lo50", "Point above hi50", "Outside"], rows),
        "",
    ]
    mem = rep.get("memory_after") or {}
    if mem and "error" not in mem:
        out.append(
            "**Memory after the run** (`docker stats`): "
            + "; ".join(f"{k} {v['mem_usage']} ({v['mem_perc']})" for k, v in mem.items())
            + "."
        )
    elif mem:
        out.append(f"**Memory after the run:** not measured ({mem['error']}).")
    env = run.get("environment") or {}
    vm, img = env.get("vm") or {}, env.get("api_container") or {}
    if vm and "error" not in vm:
        out.append(
            f"\n**Docker VM:** {vm.get('NCPU')} CPUs, "
            f"{int(vm.get('MemTotal') or 0) / 2**30:.1f} GiB, {vm.get('OperatingSystem')} "
            f"({vm.get('Architecture')}), Docker {vm.get('ServerVersion')}."
        )
    if img and "error" not in img and img.get("image_size_bytes"):
        out.append(
            f"**API image:** {int(img['image_size_bytes']) / 1e6:,.0f} MB, started "
            f"{img.get('started_at')}."
        )
    return "\n".join(out)


def _smoke_ms(by: dict[tuple[str, str], dict[str, Any]], name: str, split: str) -> str:
    c = by.get((name, split))
    return ", ".join(f"{v:.0f}" for v in c["ms"]) if c else "–"


def _smokes_section(smokes: Sequence[dict[str, Any]]) -> str:
    if not smokes:
        return "No smoke check is stored before the first run."
    rows = []
    for s in smokes:
        by = {(c["name"], c["target"].split(" ")[0]): c for c in s["checks"]}

        rows.append(
            [
                s["started_at"],
                f"`{str((s.get('build') or {}).get('git_sha'))[:7]}`",
                "pass" if s.get("passed") else "fail",
                _smoke_ms(by, "recompute", "dev"),
                _smoke_ms(by, "recompute", "holdout"),
                _smoke_ms(by, "explain", "dev"),
                _smoke_ms(by, "explain", "holdout"),
            ]
        )
    return table(
        [
            "Smoke check (UTC)",
            "Image",
            "Result",
            "Recompute dev (ms)",
            "Recompute holdout (ms)",
            "Explain dev (ms)",
            "Explain holdout (ms)",
        ],
        rows,
    )


def render(
    runs: Sequence[dict[str, Any]],
    pubs: Sequence[dict[str, Any]],
    smokes: Sequence[dict[str, Any]],
    cfg: AppConfig,
    log: Callable[[str, str], list[str]] = git_log,
) -> str:
    if not runs:
        raise ValueError("no verification run is stored: run make api-verify first")
    first = runs[0]
    rows = criteria_rows(first, pubs, cfg)
    passed = sum(r[2].startswith("**PASS**") for r in rows)
    later = len(runs) - 1
    lines = [
        "# M5: the API (Victoria)",
        "",
        "Generated by `gridcast report-m5` from the stored verification runs "
        f"(`<data root>/{cfg.serving.output_dir}/verify_*.json`, {len(runs)} stored) and the "
        "`publications` records. The criteria were pre-registered in "
        "[`docs/plans/m5_plan.md`](../docs/plans/m5_plan.md) §3 before any API code or "
        "measurement, and each is judged on the Compose service (Linux, arm64). P1 and P4 were "
        "checked by `gridcast publish`, which read every published row back; P2, P3, P5 and P6 "
        "by `gridcast api-verify` against the running container, refused unless the image was "
        "the checkout's commit and both were clean.",
        "",
        "This is a software check, not a statistical holdout: reruns are allowed, and each is "
        "shown with the change before it (§3.1). Nothing is re-scored: the served values are "
        "the scored ones.",
        "",
        "## Verdicts (the first verification run)",
        "",
        table(["#", "Pass if", "Result"], rows),
        "",
        f"**{passed} of {len(CRITERIA)} criteria pass** on the first verification run "
        f"({first['started_at']}, image `{str(first['build'].get('git_sha'))[:12]}`)"
        + (f"; {later} later run(s) are listed below." if later else ".")
        + (
            ""
            if first.get("build_unchanged", True)
            else " **The API's build changed during the run.**"
        ),
        "",
        "## Verification runs",
        "",
        _runs_section(runs, log),
        "",
        "## Latency (P5, the first run)",
        "",
        "Client-side, from sending the request to the last byte of its body, one request at a "
        "time on one keep-alive connection from the host, every timed request in one seeded "
        "random order after the warm-ups. Percentiles are numpy's default (linear).",
        "",
        _latency_section(first),
        "",
        "## Every check of the first run",
        "",
        _checks_section(first),
        "",
        "## Publications (P1, P4)",
        "",
        _publications_section(pubs),
        "",
        "## Samples (the first run)",
        "",
        _samples_section(first),
        "",
        "## Reported only (the first run)",
        "",
        _reported_section(first),
        "",
        "## Seen before the first run (informal)",
        "",
        "The step-4 smoke checks, each request's client-side time (the first is cold). The "
        "question round's cost model of explain is recorded in the plan (§4, step 5); it is "
        "an estimate, not recomputed here.",
        "",
        _smokes_section(smokes),
        "",
        "## Disclosures",
        "",
        *[f"- {d.text} ({', '.join(f'`{s}`' for s in d.sources)})" for d in DISCLOSURES],
        "- **P6 is also judged against the container** (step 5a): §3 judges every criterion on "
        "the Compose service, so `api-verify` repeats the contract checks of the test suites "
        "there, whose tests stay.",
        "",
    ]
    return "\n".join(lines)


def write_report(engine: Engine, cfg: AppConfig, path: Path = REPORT) -> Path:
    runs = load_runs(cfg.data_path(Path(cfg.serving.output_dir)))
    if not runs:
        raise ValueError("no verification run is stored: run make api-verify first")
    smokes = load_smokes(cfg.data_path(Path("logs")), runs[0]["started_at"])
    pubs = load_publications(engine, cfg.serving.region)
    path.write_text(render(runs, pubs, smokes, cfg), encoding="utf-8")
    return path
