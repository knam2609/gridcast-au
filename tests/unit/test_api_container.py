"""M5 step 4: the container's build record, its Compose files, and ``gridcast api-smoke``'s
health check (m5_plan.md §4, step 4). Recompute and explain checks run on the synthetic
walk-forward in ``test_api_models.py``."""

from __future__ import annotations

import re
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gridcast.api import smoke
from gridcast.api.app import create_app
from gridcast.api.build import build_info
from gridcast.api.catalogue import read_only_engine
from gridcast.config import PROJECT_ROOT, ServingSmokeConfig, Settings, get_settings, load_config

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


# --------------------------------------------------------------------------- build record


def test_the_image_build_arguments_win_over_git() -> None:
    s = Settings(build_git_sha=SHA, build_dirty=False)
    b = build_info(s, checkout=lambda: pytest.fail("git is not asked in the image"))
    assert (b.git_sha, b.dirty, b.source) == (SHA, False, "image")


@pytest.mark.parametrize(
    ("checkout", "want"),
    [
        (SHA, (SHA, False, "checkout")),
        (f"{SHA}-dirty", (SHA, True, "checkout")),
        (None, (None, None, "unknown")),
    ],
)
def test_without_build_arguments_git_is_asked(
    checkout: str | None, want: tuple[Any, Any, str]
) -> None:
    b = build_info(Settings(), checkout=lambda: checkout)
    assert (b.git_sha, b.dirty, b.source) == want


def test_empty_build_arguments_are_unset_and_a_bad_sha_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRIDCAST_BUILD_GIT_SHA", "")
    monkeypatch.setenv("GRIDCAST_BUILD_DIRTY", "")
    s = Settings()
    assert s.build_git_sha is None and s.build_dirty is None
    monkeypatch.setenv("GRIDCAST_BUILD_GIT_SHA", SHA)
    monkeypatch.setenv("GRIDCAST_BUILD_DIRTY", "true")
    s = Settings()
    assert s.build_git_sha == SHA and s.build_dirty is True
    with pytest.raises(ValidationError):
        Settings(build_git_sha="main")


def test_health_serves_the_build_even_when_the_database_is_down() -> None:
    engine = read_only_engine("postgresql+psycopg://nobody:x@127.0.0.1:9/none")
    client = TestClient(create_app(engine=engine), raise_server_exceptions=False)
    h = client.get("/health")
    assert h.status_code == 503
    want = build_info(get_settings())
    assert h.json()["build"] == {
        "git_sha": want.git_sha,
        "dirty": want.dirty,
        "source": want.source,
    }


# --------------------------------------------------------------------------- compose files


def _yaml(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((PROJECT_ROOT / name).read_text(encoding="utf-8"))
    return data


def test_the_api_service_is_local_and_opt_in() -> None:
    api = _yaml("docker-compose.yml")["services"]["api"]
    assert api["profiles"] == ["api"]  # `docker compose up` alone does not build it
    # No auth, so only on this machine; the host port is configurable, the container's 8000.
    assert api["ports"] == ["127.0.0.1:${GRIDCAST_API_PORT:-8000}:8000"]
    assert "volumes" not in api  # the model files come only from the override
    assert api["depends_on"]["db"]["condition"] == "service_healthy"
    assert set(api["build"]["args"]) == {"GIT_SHA", "GIT_DIRTY"}


def test_the_model_files_are_read_only_and_never_created() -> None:
    (mount,) = _yaml("docker-compose.models.yml")["services"]["api"]["volumes"]
    assert mount["type"] == "bind" and mount["read_only"] is True
    assert mount["bind"]["create_host_path"] is False  # an unmounted drive stays unmounted
    assert mount["target"] == "/data/models"


def test_the_dirty_flag_covers_every_file_the_image_copies() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    copied = set()
    for line in dockerfile.splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            copied.update(line.split()[1:-1])
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^IMAGE_FILES := (.+)$", makefile, re.MULTILINE)
    assert match is not None
    image_files = set(match.group(1).split())
    assert copied == {"pyproject.toml", "uv.lock", "README.md", "configs", "src"}
    assert copied | {"Dockerfile", ".dockerignore"} == image_files
    ignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    allowed = {line[1:].rstrip("/") for line in ignore if line.startswith("!")}
    rules = [line for line in ignore if line and not line.startswith("#")]
    assert rules[0] == "*" and allowed == copied  # nothing else, so never .env or .venv


# --------------------------------------------------------------------------- api-smoke


def _health(status: str = "ok", sha: str | None = SHA, code: int = 200) -> httpx.Client:
    body = {
        "status": status,
        "build": {"git_sha": sha, "dirty": False, "source": "image"},
        "checks": {
            "database": {"ok": True, "detail": "reachable"},
            "model_files": {"ok": status == "ok", "detail": "..."},
        },
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(code, json=body))
    return httpx.Client(transport=transport, base_url="http://api")


def test_smoke_health_passes_when_ok_and_built_from_the_checkout() -> None:
    check, build = smoke.check_health(_health(), f"{SHA}-dirty")  # the same commit, dirty
    assert check.ok and build is not None and build["git_sha"] == SHA


@pytest.mark.parametrize(
    ("client", "checkout", "message"),
    [
        (_health(status="degraded"), SHA, "failing: model_files"),
        (_health(status="unavailable", code=503), SHA, "HTTP 503"),
        (_health(sha=OTHER), SHA, "built from fedcba987654, the checkout is at 0123456789ab"),
        (_health(sha=None), SHA, "the build is unknown"),
    ],
)
def test_smoke_health_fails_with_a_reason(
    client: httpx.Client, checkout: str, message: str
) -> None:
    check, _ = smoke.check_health(client, checkout)
    assert not check.ok and message in check.detail


def test_smoke_fails_when_repeats_differ() -> None:
    bodies = iter([{"max_abs_diff_mw": 0.0}, {"max_abs_diff_mw": 1.0}])

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "split": "dev",
            "matches_stored": True,
            "leads_stored": 288,
            "leads_recomputed": 288,
            "leads_differing": 0,
            "max_float64_diff_mw": 0.0,
            **next(bodies),
        }
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api")
    t = ServingSmokeConfig(issue_times=["2024-03-15T02:00:00Z"], repeats=2).issue_times[0]
    check = smoke.check_recompute(client, t, repeats=2)
    assert not check.ok and "the repeats differ" in check.detail


def test_smoke_issue_times_are_utc_and_one_per_split() -> None:
    cfg = load_config()
    times = cfg.serving.smoke.issue_times
    holdout = cfg.backtest.holdout_start
    assert [t.date() < holdout for t in times] == [True, False]
    with pytest.raises(ValidationError, match="UTC"):
        ServingSmokeConfig(issue_times=["2024-03-15T12:00:00+10:00"], repeats=1)


def test_uvicorns_factory_configures_logging_from_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gridcast.api import app as app_module

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(app_module, "configure_logging", lambda *a: seen.append(a))
    monkeypatch.setattr(app_module, "get_settings", lambda: Settings(log_format="json"))
    assert app_module.serve().title == "GridCast-AU API"
    assert seen == [("INFO", "json")]
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "gridcast.api.app:serve" in dockerfile and "gridcast.api.app:serve" in makefile
