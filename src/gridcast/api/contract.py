"""The API's error contract as data (P6, m5_plan.md §3), shared by the tests and ``api-verify``.

* :data:`INVALID_INPUTS`: requests that must answer 422 with a message containing the given
  text, whatever is published (no database is needed to refuse them).
* :data:`SEARCH_PARAMETERS` and :data:`SEARCH_LITERALS`: the no-500 search's (route, query
  parameter) pairs and the literal values it always tries besides random text. The routes'
  other parameters depend on what is published, so each caller supplies them.
"""

from __future__ import annotations

from typing import Final

WINDOW: Final = {"start": "2025-10-01T00:00:00Z", "end": "2025-11-01T00:00:00Z"}

INVALID_INPUTS: Final[tuple[tuple[str, dict[str, str], str], ...]] = (
    ("/v1/forecasts/demand", {}, "issue_time: Field required"),
    ("/v1/forecasts/demand", {"issue_time": "2025-10-01T00:00:00"}, "offset"),
    ("/v1/forecasts/demand", {"issue_time": "2025-10-01T00:05:00Z"}, "30-minute grid"),
    ("/v1/forecasts/demand/evolution", {"target_end": "2025-10-01T00:01:00Z"}, "5-minute"),
    ("/v1/forecasts/price", {"issue_time": "noon"}, "ISO 8601"),
    ("/v1/actuals", {"start": "2025-10-01T00:00:00Z", "end": "2025-10-09T00:00:00Z"}, "at most"),
    (
        "/v1/actuals",
        {"start": "2025-10-01T00:00:00Z", "end": "2025-10-01T00:00:00Z"},
        "after start",
    ),
    ("/v1/anomalies", {**WINDOW, "tier": "huge"}, "tier"),
    ("/v1/anomalies", {**WINDOW, "limit": "0"}, "limit"),
    ("/v1/anomalies", {**WINDOW, "limit": "501"}, "limit"),
    ("/v1/anomalies", {**WINDOW, "after": "zz"}, "cursor"),
    ("/v1/forecasts/demand/recompute", {}, "issue_time: Field required"),
    ("/v1/forecasts/demand/recompute", {"issue_time": "2025-10-01T00:10:00Z"}, "grid"),
    (
        "/v1/forecasts/demand/explain",
        {"issue_time": "2025-10-01T00:00:00Z", "lead": "289"},
        "at most 288",
    ),
    ("/v1/forecasts/demand/explain", {"issue_time": "2025-10-01T00:00:00Z", "lead": "0"}, "lead"),
    ("/v1/forecasts/demand/explain", {"issue_time": "2025-10-01T00:00:00Z", "lead": "x"}, "lead"),
    ("/v1/anomalies/0", {}, "event_id"),
    ("/v1/anomalies/abc", {}, "event_id"),
    ("/v1/anomalies/99999999999999999999", {}, "event_id"),
)

SEARCH_PARAMETERS: Final[tuple[tuple[str, str], ...]] = (
    ("/v1/forecasts/demand", "issue_time"),
    ("/v1/forecasts/demand/evolution", "target_end"),
    ("/v1/forecasts/price", "issue_time"),
    ("/v1/actuals", "start"),
    ("/v1/actuals", "end"),
    ("/v1/anomalies", "start"),
    ("/v1/anomalies", "category"),
    ("/v1/anomalies", "after"),
    ("/v1/anomalies", "limit"),
    ("/v1/forecasts/demand", "region"),
    ("/v1/forecasts/demand/recompute", "issue_time"),
    ("/v1/forecasts/demand/explain", "issue_time"),
    ("/v1/forecasts/demand/explain", "lead"),
)

SEARCH_LITERALS: Final[tuple[str, ...]] = (
    "2023-08-30T14:00:00Z",
    "2025-09-30T14:00:00+00:00",
    "9999-12-31T23:30:00-23:59",
    "0001-01-01T00:00:00+14:00",
    "2023-08-30T14:00:00\x00",
    "2023-08-31T00:00:00 10:00",
)
SEARCH_EXAMPLES: Final = 300
