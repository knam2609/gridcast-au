"""PostgreSQL binary COPY of fixed-width rows, encoded and decoded with numpy.

Every published column has a fixed width (``timestamptz``, ``smallint``, ``real``,
``double precision``), so a row of the binary COPY format is a fixed-size record: a 16-bit
field count, then per field a 32-bit byte length and the value, all big-endian. Floats travel
as their IEEE bits, so a value written and read back this way is exact: no text formatting
sits in between. ``timestamptz`` is a 64-bit count of microseconds since 2000-01-01 UTC.

Timestamps are exchanged as int64 microseconds since the Unix epoch (UTC).
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from psycopg import Cursor, sql

SIGNATURE = b"PGCOPY\n\xff\r\n\x00"
HEADER = SIGNATURE + struct.pack(">ii", 0, 0)  # flags, header extension length
TRAILER = struct.pack(">h", -1)
PG_EPOCH_UNIX_US = 946_684_800 * 1_000_000  # 2000-01-01T00:00Z in Unix microseconds

# Column kind -> (big-endian numpy type, byte length on the wire).
KINDS: dict[str, tuple[str, int]] = {
    "timestamptz": (">i8", 8),
    "smallint": (">i2", 2),
    "real": (">f4", 4),
    "double": (">f8", 8),
}
NATIVE = {"timestamptz": np.int64, "smallint": np.int16, "real": np.float32, "double": np.float64}


@dataclass(frozen=True, slots=True)
class RowLayout:
    """The ordered (name, kind) columns of one fixed-width row; no column may be NULL."""

    columns: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("a row needs at least one column")
        if unknown := {k for _, k in self.columns} - set(KINDS):
            raise ValueError(f"unsupported column kinds {sorted(unknown)}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)

    @property
    def dtype(self) -> np.dtype[Any]:
        parts: list[tuple[str, str]] = [("_fields", ">i2")]
        for name, kind in self.columns:
            parts += [(f"_len_{name}", ">i4"), (name, KINDS[kind][0])]
        return np.dtype(parts)  # packed: no alignment padding, as on the wire


def encode(layout: RowLayout, values: Mapping[str, np.ndarray]) -> bytes:
    """A complete binary COPY payload (header, rows, trailer) of equal-length columns.

    Timestamps are int64 Unix microseconds. Integers must fit their column; floats are
    written with their exact bits (a float32 value in a ``double`` column is widened exactly).
    """
    lengths = {len(values[name]) for name in layout.names}
    if len(lengths) != 1:
        raise ValueError("columns of different lengths")
    rec = np.empty(lengths.pop(), dtype=layout.dtype)
    rec["_fields"] = len(layout.columns)
    for name, kind in layout.columns:
        rec[f"_len_{name}"] = KINDS[kind][1]
        column = np.asarray(values[name])
        if kind == "timestamptz":
            rec[name] = column.astype(np.int64) - PG_EPOCH_UNIX_US
        elif kind == "smallint":
            if column.size and (column.min() < -(2**15) or column.max() >= 2**15):
                raise ValueError(f"{name}: value outside smallint")
            rec[name] = column
        else:
            rec[name] = column
    return HEADER + rec.tobytes() + TRAILER


def decode(layout: RowLayout, data: bytes | bytearray | memoryview) -> dict[str, np.ndarray]:
    """Columns (native byte order) of a binary COPY payload whose rows match ``layout``.

    Raises if the payload is not binary COPY, if a row has another field count or a field
    another length (a NULL, or a column of another type), or if it is truncated.
    """
    buf = memoryview(data)
    if bytes(buf[: len(SIGNATURE)]) != SIGNATURE:
        raise ValueError("not a binary COPY payload")
    _flags, ext = struct.unpack_from(">ii", buf, len(SIGNATURE))
    start = len(SIGNATURE) + 8 + ext
    if bytes(buf[-2:]) != TRAILER:
        raise ValueError("binary COPY payload has no trailer (truncated?)")
    dtype = layout.dtype
    body = len(buf) - start - len(TRAILER)
    if body < 0 or body % dtype.itemsize:
        raise ValueError("binary COPY rows do not match the expected layout")
    rec = np.frombuffer(buf, dtype=dtype, count=body // dtype.itemsize, offset=start)
    if np.any(rec["_fields"] != len(layout.columns)):
        raise ValueError("a row has an unexpected number of fields")
    out: dict[str, np.ndarray] = {}
    for name, kind in layout.columns:
        if np.any(rec[f"_len_{name}"] != KINDS[kind][1]):
            raise ValueError(f"column {name}: NULL or unexpected width")
        column = rec[name].astype(NATIVE[kind])
        if kind == "timestamptz":
            column = column + PG_EPOCH_UNIX_US
        out[name] = column
    return out


def copy_in(
    cur: Cursor[Any], table: str, layout: RowLayout, values: Mapping[str, np.ndarray]
) -> None:
    """``COPY table (columns) FROM STDIN`` in binary format."""
    statement = sql.SQL("COPY {} ({}) FROM STDIN (FORMAT BINARY)").format(
        sql.Identifier(table), sql.SQL(", ").join(sql.Identifier(n) for n in layout.names)
    )
    with cur.copy(statement) as copy:
        copy.write(encode(layout, values))


def copy_out(cur: Cursor[Any], query: sql.Composable, layout: RowLayout) -> dict[str, np.ndarray]:
    """Rows of ``query`` (whose select list matches ``layout``) through binary COPY."""
    statement = sql.SQL("COPY ({}) TO STDOUT (FORMAT BINARY)").format(query)
    buf = bytearray()
    with cur.copy(statement) as copy:
        for block in copy:
            buf += block
    return decode(layout, buf)
