"""Polite, cached client for Open-Meteo's archive APIs.

* **Previous Runs API** (archived forecasts, variant C): hourly ``<var>_previous_day{K}``.
* **Historical Weather API** (reanalysis, variant D): hourly ``<var>`` for e.g. ``era5``.

Each request is one site and one date range. Responses are cached as JSON under
``raw/open_meteo/<api>/<model>/<site>/<start>_<end>.json`` with a ``.meta.json`` holding the
request parameters and SHA-256. A cached file is reused only if both match, so changing the
variables, coordinates or model triggers a fresh download. Paths are built, never globbed
(the data root may be an exFAT drive with AppleDouble ``._*`` files).

Responses whose last date is within ``settle_days`` of today are not cached, because the
archives fill in recent days late (ERA5 lags by about 5 days).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from gridcast.config import OpenMeteoConfig
from gridcast.ingestion.nemweb import sha256_file
from gridcast.logs import get_logger

log = get_logger(__name__)

Api = Literal["previous_runs", "archive"]


class OpenMeteoError(RuntimeError):
    """The API rejected a request (bad parameters, unknown model, ...)."""


class InvalidBodyError(RuntimeError):
    """A successful response whose body is not JSON (seen once: transient; retried)."""


@dataclass(frozen=True, slots=True)
class OpenMeteoRequest:
    api: Api
    model: str
    site_id: str
    latitude: float
    longitude: float
    hourly: tuple[str, ...]
    start_date: dt.date
    end_date: dt.date

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError(f"end_date {self.end_date} is before start_date {self.start_date}")
        if not self.hourly:
            raise ValueError("at least one hourly variable is required")

    def params(self) -> dict[str, str]:
        return {
            "latitude": f"{self.latitude:.4f}",
            "longitude": f"{self.longitude:.4f}",
            "models": self.model,
            "hourly": ",".join(self.hourly),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        }


@dataclass(frozen=True, slots=True)
class OpenMeteoResponse:
    request: OpenMeteoRequest
    payload: dict[str, Any]
    sha256: str
    n_bytes: int
    from_cache: bool
    cached_path: Path | None  # None when the range was too recent to cache

    def record(self) -> dict[str, Any]:
        """Lineage entry for ``ingestion_runs.files``."""
        return {
            "api": self.request.api,
            "model": self.request.model,
            "site_id": self.request.site_id,
            "start_date": self.request.start_date.isoformat(),
            "end_date": self.request.end_date.isoformat(),
            "file": self.cached_path.name if self.cached_path else None,
            "sha256": self.sha256,
            "bytes": self.n_bytes,
            "from_cache": self.from_cache,
            # The model grid cell Open-Meteo actually used for the requested point.
            "grid_latitude": self.payload.get("latitude"),
            "grid_longitude": self.payload.get("longitude"),
            "elevation_m": self.payload.get("elevation"),
        }


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | InvalidBodyError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


class OpenMeteoClient:
    """HTTP client with a project User-Agent, rate limiting, retries and a file cache."""

    def __init__(
        self,
        cfg: OpenMeteoConfig,
        raw_dir: Path,
        transport: httpx.BaseTransport | None = None,
        today: Callable[[], dt.date] = _utc_today,
    ) -> None:
        self.cfg = cfg
        self.cache_dir = raw_dir / "open_meteo"
        self._today = today
        self._http = httpx.Client(
            headers={"User-Agent": cfg.user_agent},
            timeout=cfg.timeout_seconds,
            follow_redirects=True,
            transport=transport,
        )
        self._last_request: float | None = None
        self._get = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential(multiplier=max(cfg.retry_backoff_seconds, 0.0), max=60),
            reraise=True,
        )(self._get_once)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def url(self, api: Api) -> str:
        return self.cfg.previous_runs_url if api == "previous_runs" else self.cfg.archive_url

    def cache_path(self, req: OpenMeteoRequest) -> Path:
        name = f"{req.start_date:%Y%m%d}_{req.end_date:%Y%m%d}.json"
        return self.cache_dir / req.api / req.model / req.site_id / name

    def is_settled(self, req: OpenMeteoRequest) -> bool:
        return req.end_date <= self._today() - dt.timedelta(days=self.cfg.settle_days)

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.cfg.request_delay_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def _get_once(self, url: str, params: dict[str, str]) -> httpx.Response:
        self._throttle()
        response = self._http.get(url, params=params)
        if response.status_code == 429 or response.status_code >= 500:
            log.warning(
                "open_meteo.retryable", status=response.status_code, body=response.text[:200]
            )
            response.raise_for_status()
        if response.status_code >= 400:
            raise OpenMeteoError(f"HTTP {response.status_code}: {_reason(response.content)}")
        try:
            json.loads(response.content)
        except ValueError as exc:
            log.warning("open_meteo.invalid_body", body=response.content[:200])
            raise InvalidBodyError(f"response body is not JSON: {response.content[:80]!r}") from exc
        return response

    def _from_cache(self, req: OpenMeteoRequest) -> OpenMeteoResponse | None:
        path = self.cache_path(req)
        meta_path = path.with_name(path.name + ".meta.json")
        if not (path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("params") != req.params():
            log.info("open_meteo.cache_params_changed", file=str(path))
            return None
        digest = sha256_file(path)
        if digest != meta.get("sha256"):
            log.warning("open_meteo.cache_corrupt", file=str(path))
            return None
        content = path.read_bytes()
        return OpenMeteoResponse(req, json.loads(content), digest, len(content), True, path)

    def fetch(self, req: OpenMeteoRequest, refresh: bool = False) -> OpenMeteoResponse:
        """Return the API response for ``req``, from the cache when possible."""
        if not refresh and (cached := self._from_cache(req)) is not None:
            return cached
        response = self._get(self.url(req.api), req.params())
        content = response.content
        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get("error"):
            raise OpenMeteoError(f"API error: {_reason(content)}")
        digest = hashlib.sha256(content).hexdigest()
        cached_path: Path | None = None
        if self.is_settled(req):
            cached_path = self.cache_path(req)
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached_path.with_name(cached_path.name + ".part")
            tmp.write_bytes(content)
            tmp.replace(cached_path)
            cached_path.with_name(cached_path.name + ".meta.json").write_text(
                json.dumps(
                    {
                        "url": self.url(req.api),
                        "params": req.params(),
                        "sha256": digest,
                        "bytes": len(content),
                        "downloaded_at": dt.datetime.now(dt.UTC).isoformat(),
                    },
                    indent=2,
                )
            )
        log.info(
            "open_meteo.downloaded",
            api=req.api,
            model=req.model,
            site=req.site_id,
            start=str(req.start_date),
            end=str(req.end_date),
            bytes=len(content),
            cached=cached_path is not None,
        )
        return OpenMeteoResponse(req, payload, digest, len(content), False, cached_path)


def _reason(content: bytes) -> str:
    try:
        body = json.loads(content)
    except ValueError:
        return content[:200].decode("utf-8", errors="replace")
    if isinstance(body, dict) and "reason" in body:
        return str(body["reason"])
    return str(body)[:200]
