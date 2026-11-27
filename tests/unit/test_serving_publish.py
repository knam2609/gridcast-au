"""M5 step 1a: the publication's pure parts (m5_plan.md §2.3): binary COPY, reshaping, P1."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from gridcast.config import ServingConfig, load_config
from gridcast.intervals.scores import rearrange_bounds
from gridcast.serving.pgcopy import HEADER, TRAILER, RowLayout, decode, encode
from gridcast.serving.publish import (
    chunks,
    compare_rows,
    long_rows,
    read_layout,
    stage_layout,
)
from gridcast.serving.sources import (
    INTERVAL_COLUMNS,
    PRODUCTS,
    Source,
    SourceMatrix,
    SourceRun,
    Window,
    exact_float32,
    read_matrix,
    read_window,
)

H = 288
STEP_US = 5 * 60 * 1_000_000
T0 = pd.Timestamp("2025-09-28 14:00", tz="UTC")


def _times(n: int, start: pd.Timestamp = T0) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="30min").as_unit("us")


def _matrix(n: int = 4, seed: int = 0, missing: int = 3) -> SourceMatrix:
    rng = np.random.default_rng(seed)
    point = (5000 + 300 * rng.standard_normal((n, H))).astype(np.float32)
    flat = rng.choice(n * H, size=missing, replace=False)
    point.ravel()[flat] = np.nan
    return SourceMatrix(_times(n), {"point": point}, 0)


def _stored(matrix: SourceMatrix, rows: np.ndarray, kind: str = "double") -> dict[str, np.ndarray]:
    """What Postgres holds after a correct publication of ``rows`` (as decoded)."""
    out = long_rows(matrix, rows)
    out["target_end"] = out["issue_time"] + out["horizon"].astype(np.int64) * STEP_US
    for c in matrix.by_column:
        out[c] = out[c].astype(np.float64 if kind == "double" else np.float32)
    return out


# --------------------------------------------------------------------------- binary COPY


def test_copy_round_trip_is_bit_exact() -> None:
    layout = RowLayout((("t", "timestamptz"), ("h", "smallint"), ("r", "real"), ("d", "double")))
    special = np.array(
        [0.0, -0.0, 1e-45, np.finfo(np.float32).max, -123.456, 5221.765625], dtype=np.float32
    )
    n = len(special)
    t = _times(n).asi8
    values = {
        "t": t,
        "h": np.arange(1, n + 1, dtype=np.int16),
        "r": special,
        "d": special.astype(np.float64) + np.array([0, 0, 0, 0, 1e-12, 0]),
    }
    back = decode(layout, encode(layout, values))
    assert np.array_equal(back["t"], t)
    assert np.array_equal(back["h"], values["h"])
    assert np.array_equal(back["r"].view(np.uint32), special.view(np.uint32))  # -0.0 kept
    assert np.array_equal(back["d"].view(np.uint64), values["d"].view(np.uint64))


def test_timestamps_use_the_postgres_epoch() -> None:
    layout = RowLayout((("t", "timestamptz"),))
    t = np.array([pd.Timestamp("2000-01-01", tz="UTC").value // 1000], dtype=np.int64)
    payload = encode(layout, {"t": t})
    (pg_us,) = struct.unpack_from(">q", payload, len(HEADER) + 2 + 4)
    assert pg_us == 0


def test_empty_payload_decodes_to_no_rows() -> None:
    layout = RowLayout((("t", "timestamptz"), ("v", "real")))
    back = decode(layout, encode(layout, {"t": np.empty(0, np.int64), "v": np.empty(0)}))
    assert len(back["t"]) == len(back["v"]) == 0


def test_decode_refuses_null_and_bad_payloads() -> None:
    layout = RowLayout((("h", "smallint"), ("v", "real")))
    null_row = struct.pack(">hih", 2, 2, 7) + struct.pack(">i", -1)  # v is NULL
    with pytest.raises(ValueError, match=r"layout|NULL|fields"):
        decode(layout, HEADER + null_row + TRAILER)
    good = encode(layout, {"h": np.array([1], np.int16), "v": np.array([1.5], np.float32)})
    with pytest.raises(ValueError, match="trailer"):
        decode(layout, good[:-2])
    with pytest.raises(ValueError, match="binary COPY"):
        decode(layout, b"h,v\n1,1.5\n")
    with pytest.raises(ValueError, match="smallint"):
        encode(layout, {"h": np.array([40_000]), "v": np.array([1.0], np.float32)})


def test_layouts_match_the_tables() -> None:
    assert stage_layout(PRODUCTS["demand_point"]).columns == (
        ("issue_time", "timestamptz"),
        ("horizon", "smallint"),
        ("point", "double"),
    )
    interval = read_layout(PRODUCTS["demand_interval"])
    assert (
        interval.names[3:] == INTERVAL_COLUMNS == ("lo95", "lo80", "lo50", "hi50", "hi80", "hi95")
    )
    assert {k for n, k in interval.columns if n in INTERVAL_COLUMNS} == {"real"}


# --------------------------------------------------------------------------- reshaping


def test_long_rows_drop_missing_pairs_in_issue_then_lead_order() -> None:
    m = _matrix(n=3, missing=5)
    rows = long_rows(m, np.arange(3))
    assert len(rows["issue_time"]) == 3 * H - 5 == m.rows_expected
    assert m.missing_pairs == 5
    order = rows["issue_time"] * 1000 + rows["horizon"]
    assert np.all(np.diff(order) > 0)
    assert rows["horizon"].min() >= 1 and rows["horizon"].max() <= H
    i = np.searchsorted(m.times_us(), rows["issue_time"])
    assert np.array_equal(m.by_column["point"][i, rows["horizon"] - 1], rows["point"])
    assert not np.isnan(rows["point"]).any()


def test_chunks_tile_the_window() -> None:
    times = pd.date_range("2025-09-29 14:00", "2025-10-02 13:30", freq="30min", tz="UTC")
    m = SourceMatrix(times, {"point": np.zeros((len(times), H), np.float32)}, 0)
    start, end = times[0] - pd.Timedelta(days=1), times[-1] + pd.Timedelta(days=1)
    parts = chunks(m, start, end)
    assert [c.label for c in parts] == ["2025-09", "2025-10"]  # NEM months
    assert parts[0].start == start and parts[-1].end == end
    assert parts[0].end == parts[1].start == pd.Timestamp("2025-09-30 14:00", tz="UTC")
    assert np.array_equal(np.concatenate([c.rows for c in parts]), np.arange(len(times)))


# --------------------------------------------------------------------------- P1


def test_a_correct_publication_is_complete() -> None:
    m = _matrix()
    rows = np.arange(len(m.times))
    res = compare_rows(m, rows, _stored(m, rows), "double", 5)
    assert res.complete and res.equal == res.rows_read == res.rows_expected == m.rows_expected


def test_one_ulp_is_a_mismatch() -> None:
    m = _matrix()
    rows = np.arange(len(m.times))
    stored = _stored(m, rows)
    stored["point"][17] = np.nextafter(stored["point"][17], np.inf)  # one float64 ulp
    res = compare_rows(m, rows, stored, "double", 5)
    assert res.value_mismatches == 1 and not res.exact and not res.complete
    assert res.examples and "value" in res.examples[0]


def test_one_float32_ulp_in_a_bound_is_a_mismatch() -> None:
    rng = np.random.default_rng(1)
    bounds = {c: rng.normal(5000, 100, (2, H)).astype(np.float32) for c in INTERVAL_COLUMNS}
    m = SourceMatrix(_times(2), bounds, 0)
    rows = np.arange(2)
    stored = _stored(m, rows, kind="real")
    stored["hi80"][3] = np.nextafter(stored["hi80"][3], np.float32(np.inf))
    res = compare_rows(m, rows, stored, "real", 5)
    assert res.value_mismatches == 1


def test_missing_unexpected_and_misplaced_rows() -> None:
    m = _matrix(n=3, missing=2)
    rows = np.arange(3)
    stored = _stored(m, rows)
    keep = np.ones(len(stored["issue_time"]), dtype=bool)
    keep[5] = False  # a stored pair goes missing
    stored = {k: v[keep] for k, v in stored.items()}
    res = compare_rows(m, rows, stored, "double", 5)
    assert res.missing_rows == 1 and res.exact and not res.complete

    # A row at a pair the source leaves missing, and one at an unknown issue time.
    i, h = np.argwhere(m.missing)[0]
    extra = {
        "issue_time": np.array([m.times_us()[i], m.times_us()[0] + 60_000_000], np.int64),
        "horizon": np.array([h + 1, 1], np.int16),
        "point": np.array([1.0, 1.0]),
    }
    extra["target_end"] = extra["issue_time"] + extra["horizon"].astype(np.int64) * STEP_US
    both = {k: np.concatenate([_stored(m, rows)[k], extra[k]]) for k in extra}
    res = compare_rows(m, rows, both, "double", 5)
    assert res.unexpected_rows == 2 and not res.exact


def test_wrong_target_end_is_caught() -> None:
    m = _matrix()
    rows = np.arange(len(m.times))
    stored = _stored(m, rows)
    stored["target_end"][0] += STEP_US
    res = compare_rows(m, rows, stored, "double", 5)
    assert res.bad_target_end == 1 and not res.exact


def test_published_matrix_rebuilds_the_stored_values() -> None:
    m = _matrix(n=2, missing=4)
    rows = np.arange(2)
    published = {"point": np.full((2, H), np.nan, np.float32)}
    compare_rows(m, rows, _stored(m, rows, kind="real"), "real", 5, published)
    a, b = published["point"], m.by_column["point"]
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(a[~np.isnan(a)].view(np.uint32), b[~np.isnan(b)].view(np.uint32))


# --------------------------------------------------------------------------- sources


def _write_wide(path: Path, times: pd.DatetimeIndex, values: np.ndarray) -> None:
    frame = pd.DataFrame(values, columns=[f"h{h:03d}" for h in range(1, values.shape[1] + 1)])
    frame.insert(0, "origin_time", times)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_read_window_restricts_and_checks_the_grid(tmp_path: Path) -> None:
    times = _times(10).delete(4)  # one gap
    path = tmp_path / "p.parquet"
    _write_wide(path, times, np.ones((9, H), np.float32))
    window = Window("dev", times[1], times[8])
    got, values, gaps = read_window(path, window, H, 30)
    assert got[0] == times[1] and got[-1] == times[7] and gaps == 1 and values.shape == (7, H)

    _write_wide(path, times + pd.Timedelta(minutes=5), np.ones((9, H), np.float32))
    with pytest.raises(ValueError, match="grid"):
        read_window(path, Window("dev", T0, T0 + pd.Timedelta(days=1)), H, 30)
    with pytest.raises(ValueError, match="expected origin_time"):
        read_window(path, window, H - 1, 30)


def test_exact_float32_refuses_wider_values() -> None:
    assert exact_float32(np.array([1.5, np.nan]), "x")[0] == np.float32(1.5)
    with pytest.raises(ValueError, match="float32"):
        exact_float32(np.array([0.1]), "x")


def test_intervals_are_rearranged_by_the_frozen_function(tmp_path: Path) -> None:
    n = 6
    rng = np.random.default_rng(3)
    centre = rng.normal(5000, 200, (n, H))
    half = {50: 100.0, 80: 200.0, 95: 300.0}
    raw = {}
    for lv, w in half.items():
        raw[f"lo{lv}"] = (centre - w + rng.normal(0, 150, (n, H))).astype(np.float32)
        raw[f"hi{lv}"] = (centre + w + rng.normal(0, 150, (n, H))).astype(np.float32)
    raw["lo80"][0, 0] = np.nan  # one missing bound: the whole pair is missing
    directory = tmp_path / "intervals" / "b"
    times = _times(n)
    for column, values in raw.items():
        _write_wide(directory / f"cqr__{column}.parquet", times, values)
    run = SourceRun(1, "b", "cqr", "interval", "demand", "VIC1", "success", "intervals/b", {})
    files = tuple(directory / f"cqr__{c}.parquet" for c in INTERVAL_COLUMNS)
    window = Window("dev", times[0], times[-1] + pd.Timedelta(minutes=30))
    src = Source(PRODUCTS["demand_interval"], window, run, files, tmp_path, "p")
    m = read_matrix(src, H, 30)

    lower = {lv: raw[f"lo{lv}"].astype(np.float64) for lv in half}
    upper = {lv: raw[f"hi{lv}"].astype(np.float64) for lv in half}
    lo_r, hi_r = rearrange_bounds(lower, upper)
    for lv in half:
        assert np.array_equal(m.by_column[f"lo{lv}"], lo_r[lv].astype(np.float32), equal_nan=True)
        assert np.array_equal(m.by_column[f"hi{lv}"], hi_r[lv].astype(np.float32), equal_nan=True)
    assert m.missing_pairs == 1 and all(np.isnan(v[0, 0]) for v in m.by_column.values())
    stacked = np.stack([m.by_column[c] for c in INTERVAL_COLUMNS])[:, ~m.missing]
    assert np.all(np.diff(stacked, axis=0) >= 0)  # nested after the sort
    assert m.extra["non_nested_before"] > 0  # the fixture crossed before


# --------------------------------------------------------------------------- config


def test_serving_config_names_every_product_and_split() -> None:
    cfg = load_config()
    sv = cfg.serving
    assert sv.sources["demand_interval"]["holdout"].run_id == 97
    data = sv.model_dump(mode="json")
    del data["sources"]["price_median_profile"]
    with pytest.raises(ValidationError, match="exactly"):
        ServingConfig.model_validate(data)
    data = sv.model_dump(mode="json")
    del data["sources"]["demand_point"]["holdout"]
    with pytest.raises(ValidationError, match="dev"):
        ServingConfig.model_validate(data)
