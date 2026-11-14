"""AEMO-flagged reference events for criterion A2 (docs/plans/m4_plan.md §3.1).

AEMO's ``DISPATCHPRICE`` carries two flags per region and interval: ``APCFLAG``, a 5-bit field
saying how the dispatch price was modified, and ``MARKETSUSPENDEDFLAG``. They are the only
anomaly labels in the data that nobody in this project chose. They are **never detector
inputs**; M4 uses them only to check that the detectors fire when the operator itself
flagged the price.

* **Class P** (price-observable by definition): an interval with any ``class_p_apc_bits``
  set, i.e. the price was at a cap, floor or administered limit.
* **Class O**: any other flagged interval (``class_o_apc_bits``, or the market suspended).
  These may leave no visible trace in Victorian price or demand.

Flagged intervals of one class whose gap is under ``merge_gap_hours`` form one event. Events
are dated by the NEM-time day of their start. The list was generated and committed in M4
step 1a, before any detector code existed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import PROJECT_ROOT, AppConfig
from gridcast.database.queries import load_market_frame
from gridcast.evaluation.report import _md_table, money
from gridcast.timeutils import NEM_TZ

# APCFLAG bit meanings, quoted from the MMS Data Model's DISPATCHPRICE page (m4_plan.md §3.1).
APC_BITS: dict[int, str] = {
    1: "APC or APF binding (ROP was outside of APC/APF)",
    2: "VoLL Override applied",
    4: "MPC or MPF binding (ROP was outside of MPC/MPF)",
    8: "Price manually overwritten",
    16: "Price Scaling via Inter-regional Loss Factor (IRLF)",
}
BITS_SOURCE = "https://mmsdocs.global-roam.com/MMS/DISPATCH/DISPATCHPRICE/"
DOC_PATH = PROJECT_ROOT / "docs" / "m4_reference_events.md"

CLASS_P = "P"
CLASS_O = "O"


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    """One AEMO-flagged event: a run of flagged intervals of one class."""

    event_id: str  # e.g. "P03": class, then its order in time within the class
    cls: str  # CLASS_P or CLASS_O
    start: pd.Timestamp  # UTC; start of the first flagged interval
    end: pd.Timestamp  # UTC; end of the last flagged interval
    n_intervals: int  # flagged intervals of this class in the event
    bits: tuple[int, ...]  # APCFLAG bits seen on those intervals
    n_suspended: int  # of which with the market suspended
    price_min: float
    price_max: float
    period: str  # "warm-up", "development", "holdout" or "after the holdout"

    @property
    def judged_by_a2(self) -> bool:
        """A2 judges class-P events in the development period (m4_plan.md §3)."""
        return self.cls == CLASS_P and self.period == "development"


def _mask(bits: list[int]) -> int:
    return reduce(lambda a, b: a | b, bits, 0)


def classify_intervals(
    apc_bits: np.ndarray, suspended: np.ndarray, p_bits: list[int], o_bits: list[int]
) -> np.ndarray:
    """``"P"``, ``"O"`` or ``""`` per interval.

    P if any class-P bit is set (it wins over everything else); otherwise O if a class-O bit
    is set or the market is suspended.
    """
    apc = np.asarray(apc_bits, dtype=np.int64)
    is_p = (apc & _mask(p_bits)) != 0
    is_o = ~is_p & (((apc & _mask(o_bits)) != 0) | (np.asarray(suspended) != 0))
    out = np.full(apc.shape, "", dtype=object)
    out[is_p] = CLASS_P
    out[is_o] = CLASS_O
    return out


def event_period(start: pd.Timestamp, cfg: AppConfig) -> str:
    """The period an event belongs to, by the NEM-time day of its start."""
    day = start.tz_convert(NEM_TZ).date()
    if day < cfg.anomalies.market_first_day:
        return "warm-up"
    if day < cfg.backtest.holdout_start:
        return "development"
    end = cfg.backtest.holdout_end
    if end is None or day < end:
        return "holdout"
    return "after the holdout"


def reference_events(market: pd.DataFrame, cfg: AppConfig) -> list[ReferenceEvent]:
    """Group a region's flagged intervals into events, P first, each class in time order.

    ``market`` has ``interval_end`` (UTC), ``rrp_aud_mwh``, ``apc_flag_bits`` and
    ``market_suspended_flag`` (as from :func:`load_market_frame`).
    """
    rules = cfg.anomalies.reference_events
    step = pd.Timedelta(minutes=cfg.nem.dispatch_interval_minutes)
    gap = pd.Timedelta(hours=rules.merge_gap_hours)
    frame = market.sort_values("interval_end").reset_index(drop=True)
    cls = classify_intervals(
        frame["apc_flag_bits"].to_numpy(),
        frame["market_suspended_flag"].to_numpy(),
        rules.class_p_apc_bits,
        rules.class_o_apc_bits,
    )
    events: list[ReferenceEvent] = []
    for label in (CLASS_P, CLASS_O):
        rows = frame.loc[cls == label]
        if rows.empty:
            continue
        ends = rows["interval_end"]
        starts = ends - step
        new = (starts - ends.shift(1)).fillna(gap) >= gap
        for number, (_, grp) in enumerate(rows.groupby(new.cumsum().to_numpy()), start=1):
            bits = int(np.bitwise_or.reduce(grp["apc_flag_bits"].to_numpy(dtype=np.int64)))
            start = grp["interval_end"].iloc[0] - step
            events.append(
                ReferenceEvent(
                    event_id=f"{label}{number:02d}",
                    cls=label,
                    start=start,
                    end=grp["interval_end"].iloc[-1],
                    n_intervals=len(grp),
                    bits=tuple(b for b in APC_BITS if bits & b),
                    n_suspended=int((grp["market_suspended_flag"] != 0).sum()),
                    price_min=float(grp["rrp_aud_mwh"].min()),
                    price_max=float(grp["rrp_aud_mwh"].max()),
                    period=event_period(start, cfg),
                )
            )
    return events


# ----------------------------------------------------------------------------- document


def _duration(delta: pd.Timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    days, rest = divmod(minutes, 24 * 60)
    hours, mins = divmod(rest, 60)
    parts = [(days, "d"), (hours, "h"), (mins, "min")]
    return " ".join(f"{n} {unit}" for n, unit in parts if n) or "0 min"


def _or_list(bits: list[int]) -> str:
    """``[1, 2, 4]`` → ``"1, 2 or 4"``."""
    words = [str(b) for b in bits]
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def _nem(ts: pd.Timestamp) -> str:
    return ts.tz_convert(NEM_TZ).strftime("%Y-%m-%d %H:%M")


def _local(ts: pd.Timestamp, tz: str) -> str:
    return ts.tz_convert(tz).strftime("%Y-%m-%d %H:%M %Z")


def _event_rows(events: list[ReferenceEvent], tz: str, with_a2: bool) -> list[list[str]]:
    rows = []
    for e in events:
        row = [
            e.event_id,
            e.period,
            _nem(e.start),
            _nem(e.end),
            _local(e.start, tz),
            _duration(e.end - e.start),
            f"{e.n_intervals:,}",
            ", ".join(str(b) for b in e.bits) or "–",
            f"{e.n_suspended:,}",
            f"{money(e.price_min, 2)} to {money(e.price_max, 2)}",
        ]
        if with_a2:
            row.append("**yes**" if e.judged_by_a2 else "no")
        rows.append(row)
    return rows


def render_reference_events(events: list[ReferenceEvent], cfg: AppConfig, region_id: str) -> str:
    """Markdown for ``docs/m4_reference_events.md`` (deterministic: no run timestamps)."""
    rules = cfg.anomalies.reference_events
    tz = cfg.regions[region_id].local_tz
    holdout_end = cfg.backtest.holdout_end
    last_holdout_day = (
        (holdout_end - dt.timedelta(days=1)).isoformat() if holdout_end else "the last loaded day"
    )
    dev_last = (cfg.backtest.holdout_start - dt.timedelta(days=1)).isoformat()
    p_bits = _or_list(rules.class_p_apc_bits)
    o_bits = _or_list(rules.class_o_apc_bits)
    judged = [e for e in events if e.judged_by_a2]

    def bit_class(bit: int) -> str:
        if bit in rules.class_p_apc_bits:
            return CLASS_P
        return CLASS_O if bit in rules.class_o_apc_bits else "–"

    headers = [
        "Event",
        "Period",
        "Start (NEM time)",
        "End (NEM time)",
        "Local start",
        "Duration",
        "Flagged intervals",
        "APCFLAG bits",
        "Suspended intervals",
        "Price range on flagged intervals",
    ]
    out: list[str] = [
        "# M4 reference events: AEMO-flagged intervals",
        "",
        "> **Generated** by `gridcast anomaly-reference-events` from the stored `DISPATCHPRICE` "
        f"flags of {region_id} (`market_intervals.apc_flag_bits`, `market_suspended_flag`). "
        "Do not edit by hand. Generated and committed in M4 step 1a, **before any detector "
        "code existed**. The rules are in [`docs/plans/m4_plan.md`](plans/m4_plan.md) §3.1 and "
        "`configs/base.yaml` (`anomalies.reference_events`).",
        "",
        "These flags are AEMO's own record that it modified a dispatch price or suspended the "
        "market. They are the only anomaly labels in the data that nobody in this project "
        "chose. **They are never detector inputs.** Criterion A2 uses them only to check that "
        "the detectors fire when the operator flagged the price.",
        "",
        "## What the flags mean",
        "",
        '`APCFLAG` is "a 5-bit Region-based field indicating that the original Dispatch Price '
        "(ROP) calculated by the Dispatch Algorithm for a region has undergone modification by "
        'one of [sic] more of the following processes" (MMS Data Model, `DISPATCHPRICE`):',
        "",
        _md_table(
            ["Bit value", "Meaning (quoted)", "Class"],
            [[str(b), text, bit_class(b)] for b, text in APC_BITS.items()],
            align="lll",
        ),
        "",
        "- MPC/MPF: market price cap/floor. APC/APF: administered price cap/floor. "
        "ROP: the regional price before any capping, scaling or override.",
        '- `MARKETSUSPENDEDFLAG`: "Market suspended flag"; the page gives no further '
        "description. Class O.",
        f"- Source: the DISPATCHPRICE page of Global-Roam's mirror of AEMO's MMS Data Model, "
        f"<{BITS_SOURCE}>, read on 2026-11-15. AEMO's own Data Model pages returned HTTP 404 "
        "that day. The bit meanings match the comment in `src/gridcast/ingestion/dispatch.py` "
        "that the plan was written from, so the classes needed no change.",
        "",
        "## Rules",
        "",
        f"- **Class P** (price-observable by definition): an interval with APCFLAG bit "
        f"{p_bits} set, i.e. the price was at a cap, floor or administered limit. "
        "Class P wins when an interval also has other flags.",
        f"- **Class O**: any other flagged interval, i.e. APCFLAG bit {o_bits} only, or the "
        "market suspended. These may leave no visible trace in Victorian price or demand, "
        "so their recall is reported with no claim that anything observable happened.",
        f"- **Events:** flagged intervals of one class whose gap is under "
        f"{rules.merge_gap_hours:g} h form one event. It starts at the start of its first "
        "flagged interval and ends at the end of its last.",
        f"- **Periods**, by the NEM-time day of the event's start: warm-up before "
        f"{cfg.anomalies.market_first_day.isoformat()} (never scored); development "
        f"{cfg.anomalies.market_first_day.isoformat()} → {dev_last}; holdout "
        f"{cfg.backtest.holdout_start.isoformat()} → {last_holdout_day}.",
        f"- **A2** judges every class-P event in development: it passes if each has at least "
        f"one notable alert from a calibrated detector (the fixed price lines excluded) "
        f"between {rules.alert_margin_minutes} min before its start and "
        f"{rules.alert_margin_minutes} min after its end.",
        "",
        "## Summary",
        "",
    ]
    periods = ["warm-up", "development", "holdout", "after the holdout"]
    out += [
        _md_table(
            ["Class", *periods, "Total"],
            [
                [
                    label,
                    *[str(sum(e.cls == label and e.period == p for e in events)) for p in periods],
                    str(sum(e.cls == label for e in events)),
                ]
                for label in (CLASS_P, CLASS_O)
            ],
        ),
        "",
        f"**A2 judges {len(judged)} class-P events:** "
        + (", ".join(e.event_id for e in judged) or "none")
        + ". The holdout has "
        + (
            "no class-P event, so A2 is judged on development only."
            if not any(e.cls == CLASS_P and e.period == "holdout" for e in events)
            else "class-P events too; they are reported, not judged (m4_plan.md §3.2)."
        ),
        "",
        "## Class P events",
        "",
        _md_table(
            [*headers, "Judged by A2"],
            _event_rows([e for e in events if e.cls == CLASS_P], tz, with_a2=True),
            align="llllllrlrll",
        ),
        "",
        "## Class O events",
        "",
        _md_table(
            headers,
            _event_rows([e for e in events if e.cls == CLASS_O], tz, with_a2=False),
            align="llllllrlrl",
        ),
        "",
    ]
    return "\n".join(out)


def write_reference_events(
    engine: Engine, cfg: AppConfig, region_id: str, path: Path = DOC_PATH
) -> list[ReferenceEvent]:
    """Build the reference events from the stored flags, write the document, return them."""
    events = reference_events(load_market_frame(engine, region_id), cfg)
    path.write_text(render_reference_events(events, cfg, region_id), encoding="utf-8")
    return events
