"""Query inputs: timestamps, grids and the anomaly cursor (m5_plan.md §1, §2.4).

Every timestamp in is ISO 8601 with an explicit UTC offset (``Z``, ``+10:00`` …); one without
is refused. A ``+`` in a query string decodes to a space unless it is percent-encoded
(``%2B``), so a space where the offset sign belongs is read as ``+``. Invalid input raises
:class:`InputError` (HTTP 422) with a message; nothing a client sends reaches the database
unchecked.
"""

from __future__ import annotations

import base64
import binascii
import datetime as dt
import re

import pandas as pd

EPOCH = pd.Timestamp(0, tz="UTC").as_unit("us")
# Date, then a time with seconds or minutes, then a space where a "+" offset was decoded.
_SPACE_OFFSET = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.,]+) (\d{2}(:?\d{2})?)$")
CURSOR_VERSION = "v1"
MAX_ID = 2**63 - 1


class InputError(ValueError):
    """An invalid query input: HTTP 422 with this message."""


class NotPublished(LookupError):
    """Valid input, nothing published there: HTTP 404 with this message."""


class Unavailable(RuntimeError):
    """A dependency is down or a product is not published: HTTP 503 with this message."""


def parse_instant(name: str, value: str) -> pd.Timestamp:
    """An ISO 8601 timestamp with an offset, as a UTC timestamp (microseconds)."""
    text = value.strip()
    if m := _SPACE_OFFSET.match(text):
        text = f"{m.group(1)}+{m.group(2)}"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{name}: not an ISO 8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InputError(f"{name}: a UTC offset is required (e.g. 'Z' or '+10:00'): {value!r}")
    try:
        return pd.Timestamp(parsed.astimezone(dt.UTC)).as_unit("us")
    except (OverflowError, ValueError) as exc:
        raise InputError(f"{name}: timestamp out of range: {value!r}") from exc


def check_grid(name: str, ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    """``ts`` unchanged if it lies on the ``minutes`` grid (from the Unix epoch, in UTC)."""
    if (ts - EPOCH) % pd.Timedelta(minutes=minutes) != pd.Timedelta(0):
        raise InputError(f"{name}: must lie on the {minutes}-minute grid, got {ts.isoformat()}")
    return ts


def check_range(start: pd.Timestamp, end: pd.Timestamp, max_span: pd.Timedelta | None) -> None:
    if end <= start:
        raise InputError("end must be after start")
    if max_span is not None and end - start > max_span:
        raise InputError(f"the range may span at most {max_span}")


def check_choice(name: str, value: str | None, allowed: tuple[str, ...]) -> str | None:
    if value is not None and value not in allowed:
        raise InputError(f"{name}: unknown value {value!r}; one of {list(allowed)}")
    return value


def encode_cursor(first_interval_end: pd.Timestamp, event_id: int) -> str:
    """Opaque keyset cursor of an anomaly event: its (first interval end, event id)."""
    us = (first_interval_end.as_unit("us") - EPOCH) // pd.Timedelta(microseconds=1)
    raw = f"{CURSOR_VERSION}:{int(us)}:{int(event_id)}".encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(value: str) -> tuple[pd.Timestamp, int]:
    """The (first interval end, event id) a cursor encodes; :class:`InputError` otherwise."""
    bad = InputError(f"after: not a cursor returned by this API: {value!r}")
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("ascii")
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise bad from exc
    parts = raw.split(":")
    if len(parts) != 3 or parts[0] != CURSOR_VERSION:
        raise bad
    try:
        us, event_id = int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise bad from exc
    if not (0 < event_id <= MAX_ID) or abs(us) > 10**17:
        raise bad
    try:
        return EPOCH + pd.Timedelta(microseconds=us), event_id
    except (OverflowError, ValueError) as exc:
        raise bad from exc
