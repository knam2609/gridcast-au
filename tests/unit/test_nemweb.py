import json
from pathlib import Path

import httpx
import pytest

from gridcast.config import MmsdmSourceConfig
from gridcast.ingestion.nemweb import NemwebClient, month_dir_url, parse_listing

BASE = "https://nemweb.example/MMSDM"
PAGE_2025 = month_dir_url(BASE, 2025, 1)
PATH_2025 = "/MMSDM/2025/MMSDM_2025_01/MMSDM_Historical_Data_SQLLoader/DATA/"

LISTING_NEW = f"""<html><body><pre>
<A HREF="{PATH_2025}PUBLIC_ARCHIVE%23DISPATCHPRICE%23FILE01%23202501010000.zip">x</A>
<A HREF="{PATH_2025}PUBLIC_ARCHIVE%23DISPATCHREGIONSUM%23FILE01%23202501010000.zip">x</A>
<A HREF="{PATH_2025}PUBLIC_ARCHIVE%23DISPATCHREGIONSUM%23FILE02%23202501010000.zip">x</A>
<A HREF="{PATH_2025}PUBLIC_ARCHIVE%23PREDISPATCHREGIONSUM%23FILE01%23202501010000.zip">x</A>
<A HREF="{PATH_2025}PUBLIC_ARCHIVE%23PREDISPATCHPRICE%23FILE01%23202501010000.zip">x</A>
</pre></body></html>"""

LISTING_OLD = """
<A HREF="/x/PUBLIC_DVD_DISPATCHREGIONSUM_202110010000.zip">a</A>
<A HREF="/x/PUBLIC_DVD_PREDISPATCHREGIONSUM_D_202110010000.zip">b</A>
<A HREF="/x/PUBLIC_DVD_DISPATCHPRICE_202110010000.zip">c</A>
"""


def _cfg(**kw: object) -> MmsdmSourceConfig:
    base = {
        "base_url": BASE,
        "user_agent": "gridcast-test",
        "request_delay_seconds": 0.0,
        "timeout_seconds": 5,
        "max_retries": 3,
        "tables": ["DISPATCHREGIONSUM"],
    }
    return MmsdmSourceConfig.model_validate(base | kw)


def test_parse_listing_new_format_multipart_and_excludes_predispatch() -> None:
    files = parse_listing(LISTING_NEW, "DISPATCHREGIONSUM", 2025, 1, PAGE_2025)
    assert [f.part for f in files] == [1, 2]
    assert all(f.filename.startswith("PUBLIC_ARCHIVE#DISPATCHREGIONSUM#") for f in files)
    assert files[0].url.endswith("PUBLIC_ARCHIVE%23DISPATCHREGIONSUM%23FILE01%23202501010000.zip")
    price = parse_listing(LISTING_NEW, "DISPATCHPRICE", 2025, 1, PAGE_2025)
    assert [f.filename for f in price] == ["PUBLIC_ARCHIVE#DISPATCHPRICE#FILE01#202501010000.zip"]


def test_parse_listing_old_format() -> None:
    page = "https://nemweb.example/x/"
    files = parse_listing(LISTING_OLD, "DISPATCHREGIONSUM", 2021, 10, page)
    assert [f.filename for f in files] == ["PUBLIC_DVD_DISPATCHREGIONSUM_202110010000.zip"]
    assert files[0].part == 1


def test_parse_listing_rejects_wrong_month() -> None:
    assert parse_listing(LISTING_OLD, "DISPATCHREGIONSUM", 2021, 11, "https://n.example/") == []


def _transport(calls: list[str], fail_first: int = 0) -> httpx.MockTransport:
    state = {"failures": fail_first}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert request.headers["user-agent"] == "gridcast-test"
        if state["failures"] > 0:
            state["failures"] -= 1
            return httpx.Response(503)
        if str(request.url) == PAGE_2025:
            return httpx.Response(200, text=LISTING_NEW)
        return httpx.Response(200, content=b"zip-bytes:" + request.url.path.encode())

    return httpx.MockTransport(handler)


def test_fetch_month_downloads_then_uses_cache(tmp_path: Path) -> None:
    calls: list[str] = []
    with NemwebClient(_cfg(), tmp_path, transport=_transport(calls)) as client:
        first = client.fetch_month("DISPATCHREGIONSUM", 2025, 1)
        assert [d.from_cache for d in first] == [False, False]
        n_calls = len(calls)  # 1 listing + 2 parts
        assert n_calls == 3
        second = client.fetch_month("DISPATCHREGIONSUM", 2025, 1)
        assert [d.from_cache for d in second] == [True, True]
        assert len(calls) == n_calls  # nothing re-downloaded, listing cached
    meta = json.loads(first[0].path.with_name(first[0].path.name + ".meta.json").read_text())
    assert meta["sha256"] == first[0].sha256


def test_corrupted_cache_is_redownloaded(tmp_path: Path) -> None:
    calls: list[str] = []
    with NemwebClient(_cfg(), tmp_path, transport=_transport(calls)) as client:
        first = client.fetch_month("DISPATCHREGIONSUM", 2025, 1)
        first[0].path.write_bytes(b"corrupted")
        again = client.fetch_month("DISPATCHREGIONSUM", 2025, 1)
        assert [d.from_cache for d in again] == [False, True]
        assert again[0].sha256 == first[0].sha256


def test_retries_transient_server_errors(tmp_path: Path) -> None:
    calls: list[str] = []
    with NemwebClient(_cfg(), tmp_path, transport=_transport(calls, fail_first=2)) as client:
        files = client.list_month("DISPATCHREGIONSUM", 2025, 1)
    assert len(files) == 2
    assert len(calls) == 3


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    calls: list[str] = []
    with (
        NemwebClient(_cfg(), tmp_path, transport=_transport(calls, fail_first=10)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        client.list_month("DISPATCHREGIONSUM", 2025, 1)
    assert len(calls) == 3


PD_PAGE = month_dir_url(BASE, 2025, 1, "PREDISP_ALL_DATA")
PD_PATH = "/MMSDM/2025/MMSDM_2025_01/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/"
LISTING_ALL = f"""<pre>
<A HREF="{PD_PATH}PUBLIC_ARCHIVE%23PREDISPATCHREGIONSUM%23ALL%23FILE01%23202501010000.zip">a</A>
<A HREF="{PD_PATH}PUBLIC_ARCHIVE%23PREDISPATCHPRICE%23ALL%23FILE01%23202501010000.zip">b</A>
<A HREF="{PD_PATH}PUBLIC_ARCHIVE%23PREDISPATCHREGIONSUM%23FILE01%23202501010000.zip">c</A>
</pre>"""


def test_all_runs_listing_matches_only_the_all_archive() -> None:
    assert PD_PAGE.endswith("/MMSDM_2025_01/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/")
    files = parse_listing(
        LISTING_ALL, "PREDISPATCHREGIONSUM", 2025, 1, PD_PAGE, "PREDISP_ALL_DATA", all_runs=True
    )
    assert [f.filename for f in files] == [
        "PUBLIC_ARCHIVE#PREDISPATCHREGIONSUM#ALL#FILE01#202501010000.zip"
    ]
    assert files[0].folder == "PREDISP_ALL_DATA" and files[0].all_runs
    assert files[0].dataset == "PREDISPATCHREGIONSUM_ALL"
    # The DATA/ pattern never picks up an #ALL# archive (and vice versa).
    plain = parse_listing(LISTING_ALL, "PREDISPATCHREGIONSUM", 2025, 1, PD_PAGE)
    assert [f.filename for f in plain] == [
        "PUBLIC_ARCHIVE#PREDISPATCHREGIONSUM#FILE01#202501010000.zip"
    ]


def test_all_runs_archive_cached_apart_from_data_table(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url) == PD_PAGE:
            return httpx.Response(200, text=LISTING_ALL)
        return httpx.Response(200, content=b"zip")

    with NemwebClient(_cfg(), tmp_path, transport=httpx.MockTransport(handler)) as client:
        got = client.fetch_month(
            "PREDISPATCHREGIONSUM", 2025, 1, folder="PREDISP_ALL_DATA", all_runs=True
        )
        again = client.fetch_month(
            "PREDISPATCHREGIONSUM", 2025, 1, folder="PREDISP_ALL_DATA", all_runs=True
        )
    assert [d.from_cache for d in again] == [True]
    assert got[0].path.parent == tmp_path / "mmsdm" / "PREDISPATCHREGIONSUM_ALL" / "202501"
    assert (got[0].path.parent / "listing.json").exists()
    assert len(calls) == 2  # one listing, one file
