"""Polite, cached downloader for AEMO's MMSDM monthly archive on NEMWeb.

Files are discovered from the month's directory listing (not guessed), because AEMO's
file naming changed over time:

* ``PUBLIC_DVD_DISPATCHREGIONSUM_202110010000.zip``            (older)
* ``PUBLIC_ARCHIVE#DISPATCHREGIONSUM#FILE01#202501010000.zip`` (newer, may have FILE02…)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Self
from urllib.parse import unquote, urljoin

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from gridcast.config import MmsdmSourceConfig
from gridcast.logs import get_logger

log = get_logger(__name__)

_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ArchiveFile:
    """One downloadable MMSDM table file for one month."""

    table: str
    year: int
    month: int
    part: int
    filename: str  # decoded, e.g. PUBLIC_ARCHIVE#DISPATCHPRICE#FILE01#202501010000.zip
    url: str


@dataclass(frozen=True, slots=True)
class DownloadedFile:
    file: ArchiveFile
    path: Path
    sha256: str
    n_bytes: int
    from_cache: bool


def month_dir_url(base_url: str, year: int, month: int) -> str:
    return (
        f"{base_url.rstrip('/')}/{year:04d}/MMSDM_{year:04d}_{month:02d}/"
        "MMSDM_Historical_Data_SQLLoader/DATA/"
    )


def table_file_pattern(table: str) -> re.Pattern[str]:
    """Anchored pattern matching only ``table`` (never e.g. ``PRE<table>`` or ``<table>_D``)."""
    t = re.escape(table)
    return re.compile(
        rf"^PUBLIC_(?:"
        rf"DVD_{t}(?:_FILE(?P<part_dvd>\d{{2}}))?_(?P<stamp_dvd>\d{{12}})"
        rf"|ARCHIVE#{t}#FILE(?P<part_arc>\d{{2}})#(?P<stamp_arc>\d{{12}})"
        rf")\.zip$"
    )


def parse_listing(html: str, table: str, year: int, month: int, page_url: str) -> list[ArchiveFile]:
    """Extract the archive files for ``table`` from a NEMWeb directory-listing page."""
    pattern = table_file_pattern(table)
    expected_stamp = f"{year:04d}{month:02d}010000"
    found: dict[str, ArchiveFile] = {}
    for href in _HREF_RE.findall(html):
        filename = unquote(href.rsplit("/", 1)[-1])
        m = pattern.match(filename)
        if not m:
            continue
        stamp = m.group("stamp_dvd") or m.group("stamp_arc")
        if stamp != expected_stamp:
            log.warning("mmsdm.unexpected_stamp", file=filename, expected=expected_stamp)
            continue
        part = int(m.group("part_dvd") or m.group("part_arc") or 1)
        found[filename] = ArchiveFile(
            table=table,
            year=year,
            month=month,
            part=part,
            filename=filename,
            url=urljoin(page_url, href),
        )
    return sorted(found.values(), key=lambda f: f.part)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class NemwebClient:
    """HTTP client for NEMWeb with retries, rate limiting and a local file cache."""

    def __init__(
        self,
        cfg: MmsdmSourceConfig,
        raw_dir: Path,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cfg = cfg
        self.raw_dir = raw_dir
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
            wait=wait_exponential(multiplier=max(cfg.request_delay_seconds, 0.01), max=60),
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

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.cfg.request_delay_seconds - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
        self._last_request = time.monotonic()

    def _get_once(self, url: str) -> httpx.Response:
        self._throttle()
        response = self._http.get(url)
        response.raise_for_status()
        return response

    def list_month(self, table: str, year: int, month: int) -> list[ArchiveFile]:
        page_url = month_dir_url(self.cfg.base_url, year, month)
        html = self._get(page_url).text
        files = parse_listing(html, table, year, month, page_url)
        if not files:
            raise FileNotFoundError(f"No {table} file listed for {year}-{month:02d} at {page_url}")
        return files

    def cache_path(self, f: ArchiveFile) -> Path:
        return self.raw_dir / "mmsdm" / f.table / f"{f.year:04d}{f.month:02d}" / f.filename

    def download(self, f: ArchiveFile) -> DownloadedFile:
        """Download ``f`` into the cache unless an intact cached copy already exists."""
        path = self.cache_path(f)
        meta_path = path.with_name(path.name + ".meta.json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            digest = sha256_file(path)
            if digest == meta.get("sha256"):
                return DownloadedFile(f, path, digest, path.stat().st_size, from_cache=True)
            log.warning("mmsdm.cache_corrupt", file=str(path))

        path.parent.mkdir(parents=True, exist_ok=True)
        response = self._get(f.url)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(response.content)
        digest = hashlib.sha256(response.content).hexdigest()
        tmp.replace(path)
        meta_path.write_text(
            json.dumps(
                {
                    "url": f.url,
                    "sha256": digest,
                    "bytes": len(response.content),
                    "downloaded_at": dt.datetime.now(dt.UTC).isoformat(),
                },
                indent=2,
            )
        )
        log.info("mmsdm.downloaded", file=f.filename, bytes=len(response.content))
        return DownloadedFile(f, path, digest, len(response.content), from_cache=False)

    def month_files(self, table: str, year: int, month: int) -> list[ArchiveFile]:
        """Files for ``table``/month, from a cached listing if available, else NEMWeb.

        The listing itself is cached so that multi-part months can never be partially
        reused from cache (every listed part is always downloaded or hash-verified).
        """
        listing_path = self.raw_dir / "mmsdm" / table / f"{year:04d}{month:02d}" / "listing.json"
        if listing_path.exists():
            return [ArchiveFile(**item) for item in json.loads(listing_path.read_text())]
        files = self.list_month(table, year, month)
        listing_path.parent.mkdir(parents=True, exist_ok=True)
        listing_path.write_text(json.dumps([asdict(f) for f in files], indent=2))
        return files

    def fetch_month(self, table: str, year: int, month: int) -> list[DownloadedFile]:
        """Download (or reuse from cache) every part of ``table`` for one month."""
        return [self.download(f) for f in self.month_files(table, year, month)]
