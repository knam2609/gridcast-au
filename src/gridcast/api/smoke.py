"""``gridcast api-smoke``: a quick check of a running API (m5_plan.md §4, step 4).

``/health`` must be ``ok``, and the API must be built from the checkout's commit. Then, at each
of ``serving.smoke.issue_times``, recompute must match the stored forecast on every lead and
explain must pass E1's gate. Each request is made ``serving.smoke.repeats`` times (the first
is cold), and the repeats must return the same body. Informal: P2, P3 and P5 are judged by
``api-verify`` in step 5.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from gridcast.config import AppConfig, ServingSmokeConfig

RECOMPUTE = "/v1/forecasts/demand/recompute"
EXPLAIN = "/v1/forecasts/demand/explain"


@dataclass(slots=True)
class Check:
    name: str
    target: str  # what was asked: "/health", or the split and issue time
    ok: bool
    detail: str
    ms: list[float] = field(default_factory=list)  # client-side, in request order
    kib: float | None = None  # response body size


@dataclass(slots=True)
class SmokeResult:
    url: str
    started_at: str
    checkout_git_sha: str | None
    build: dict[str, Any] | None
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(client: httpx.Client, path: str, params: dict[str, str] | None) -> tuple[Any, float]:
    t0 = time.perf_counter()
    r = client.get(path, params=params)
    return r, (time.perf_counter() - t0) * 1000


def _body(r: httpx.Response) -> dict[str, Any]:
    try:
        body = r.json()
    except ValueError:
        return {"detail": r.text[:200]}
    return body if isinstance(body, dict) else {"detail": str(body)[:200]}


def check_health(client: httpx.Client, checkout: str | None) -> tuple[Check, dict[str, Any] | None]:
    """``/health`` is ``ok``, and the API runs the checkout's commit (``checkout`` is
    :func:`~gridcast.provenance.git_sha`'s ``<sha>[-dirty]``, or None when unknown)."""
    head = checkout.removesuffix("-dirty") if checkout else None
    r, ms = _get(client, "/health", None)
    body = _body(r)
    problems = []
    if r.status_code != 200 or body.get("status") != "ok":
        failing = [k for k, c in (body.get("checks") or {}).items() if not c.get("ok")]
        problems.append(
            f"HTTP {r.status_code}, status {body.get('status')!r}"
            + (f", failing: {', '.join(failing)}" if failing else "")
        )
    build = body.get("build")
    sha = build.get("git_sha") if isinstance(build, dict) else None
    if sha is None:
        problems.append("the build is unknown")
    elif head is not None and sha != head:
        problems.append(f"built from {sha[:12]}, the checkout is at {head[:12]}")
    detail = "; ".join(problems) or "ok, every check passed"
    return Check("health", "/health", not problems, detail, [ms]), build


def _repeat(
    client: httpx.Client, path: str, params: dict[str, str], repeats: int
) -> tuple[list[httpx.Response], list[float]]:
    out = [_get(client, path, params) for _ in range(repeats)]
    return [r for r, _ in out], [ms for _, ms in out]


def _same(responses: list[httpx.Response]) -> bool:
    return all(r.content == responses[0].content for r in responses[1:])


def check_recompute(client: httpx.Client, issue_time: dt.datetime, repeats: int) -> Check:
    """Every lead recomputed equals the stored float32 (gaps included), on every repeat."""
    t = iso(issue_time)
    responses, ms = _repeat(client, RECOMPUTE, {"issue_time": t}, repeats)
    r = responses[0]
    b = _body(r)
    if r.status_code != 200:
        return Check("recompute", t, False, f"HTTP {r.status_code}: {b.get('detail')}", ms)
    target = f"{b['split']} {t}"
    ok = bool(b["matches_stored"]) and b["leads_differing"] == 0 and _same(responses)
    detail = (
        f"{b['leads_recomputed']}/{b['leads_stored']} leads, {b['leads_differing']} differ, "
        f"max |diff| {b['max_abs_diff_mw']} MW (float64 {b['max_float64_diff_mw']:.3g} MW)"
    )
    if not _same(responses):
        detail += "; the repeats differ"
    return Check("recompute", target, ok, detail, ms, len(r.content) / 1024)


def check_explain(client: httpx.Client, issue_time: dt.datetime, repeats: int) -> Check:
    """Every stored lead explained, E1's gate passed, on every repeat."""
    t = iso(issue_time)
    responses, ms = _repeat(client, EXPLAIN, {"issue_time": t}, repeats)
    r = responses[0]
    b = _body(r)
    gate = b.get("gate") or {}
    if r.status_code != 200:
        detail = f"HTTP {r.status_code}: {b.get('detail')}"
        if gate:
            detail += f" (gate: {gate})"
        return Check("explain", t, False, detail, ms)
    target = f"{b['split']} {t}"
    ok = bool(gate["passed"]) and _same(responses)
    detail = (
        f"{gate['n']} leads explained, {len(b['leads_without_forecast'])} without a forecast; "
        f"gate {'passed' if gate['passed'] else 'FAILED'} ({gate['forecast_mismatches']} "
        f"forecast mismatches, max sum error {gate['max_sum_error']:.3g} "
        f"≤ {gate['tolerance']:g})"
    )
    if not _same(responses):
        detail += "; the repeats differ"
    return Check("explain", target, ok, detail, ms, len(r.content) / 1024)


def run_smoke(
    client: httpx.Client, smoke: ServingSmokeConfig, checkout: str | None, url: str
) -> SmokeResult:
    started = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    health, build = check_health(client, checkout)
    checks = [health]
    for t in smoke.issue_times:
        checks.append(check_recompute(client, t, smoke.repeats))
        checks.append(check_explain(client, t, smoke.repeats))
    return SmokeResult(url, started, checkout, build, checks)


def render(result: SmokeResult) -> str:
    b = result.build or {}
    state = {True: "dirty", False: "clean", None: "dirty unknown"}[b.get("dirty")]
    lines = [
        f"api-smoke {result.url} at {result.started_at}: " + ("PASS" if result.passed else "FAIL"),
        f"build: {b.get('git_sha')} ({b.get('source')}, {state}); "
        f"checkout: {result.checkout_git_sha}",
    ]
    for c in result.checks:
        times = ", ".join(f"{v:.0f}" for v in c.ms)
        size = f", {c.kib:,.0f} KiB" if c.kib is not None else ""
        lines.append(
            f"  {'ok  ' if c.ok else 'FAIL'} {c.name:<9} {c.target:<30} {c.detail} "
            f"[ms: {times}{size}]"
        )
    return "\n".join(lines)


def save(result: SmokeResult, cfg: AppConfig) -> Path:
    """The result as JSON under ``<data root>/logs`` (beside the publish logs)."""
    stamp = result.started_at.replace("-", "").replace(":", "").replace("T", "_").rstrip("Z")
    path = cfg.data_path(Path("logs")) / f"api_smoke_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(result), "passed": result.passed}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
