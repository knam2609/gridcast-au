"""Weather ETL: Open-Meteo responses → validated hourly frames → PostgreSQL.

Work is chunked by **NEM month**, like the MMSDM ingest: one ``ingestion_runs`` row per
month and dataset (``weather_forecasts`` = variant C, ``weather_observations`` = variant D),
each covering every configured site. A month needs the hours from its first interval's
``floor_1h`` to its last interval's ``ceil_1h``, i.e. ``[month start, next month start]`` in
UTC. Adjacent months therefore share one hour, which the idempotent upsert absorbs.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import pandera.errors as pe
from sqlalchemy import Engine

from gridcast.config import WEATHER_VARIABLES, AppConfig, WeatherVariableConfig
from gridcast.database.load import (
    ensure_regions,
    ensure_weather_sites,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_weather_forecasts,
    upsert_weather_observations,
)
from gridcast.ingestion.open_meteo import OpenMeteoClient, OpenMeteoRequest, OpenMeteoResponse
from gridcast.ingestion.pipeline import iter_months
from gridcast.logs import get_logger
from gridcast.timeutils import month_hours_utc
from gridcast.validation.schemas import (
    summarise_schema_errors,
    weather_forecasts_schema,
    weather_observations_schema,
)
from gridcast.weather.runs import available_at_for, run_init_for

log = get_logger(__name__)

SOURCE = "open_meteo"
Kind = Literal["forecasts", "observations"]
DATASETS: dict[Kind, str] = {
    "forecasts": "weather_forecasts",
    "observations": "weather_observations",
}


@dataclass(frozen=True, slots=True)
class WeatherMonthResult:
    period: dt.date
    dataset: str
    status: str
    rows_parsed: int = 0
    rows_loaded: int = 0
    error: str | None = None


def month_hours(period: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last valid hour (UTC, inclusive) needed for a NEM month's intervals."""
    return month_hours_utc(period.year, period.month)


def forecast_column(variable: str, offset_days: int) -> str:
    return f"{variable}_previous_day{offset_days}"


def hourly_frame(payload: Mapping[str, Any], units: Mapping[str, str]) -> pd.DataFrame:
    """Hourly block of an Open-Meteo response as a frame with a UTC ``valid_time`` column.

    ``units`` maps every expected column to the unit Open-Meteo must report for it; a missing
    column or a different unit raises instead of being silently misread.
    """
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError(f"expected GMT/UTC times, got offset {payload.get('utc_offset_seconds')}")
    hourly = payload.get("hourly")
    reported = payload.get("hourly_units") or {}
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("response has no hourly block")
    missing = [c for c in units if c not in hourly]
    if missing:
        raise ValueError(f"response lacks hourly columns {missing}")
    wrong = {c: reported.get(c) for c, u in units.items() if reported.get(c) != u}
    if wrong:
        raise ValueError(f"unexpected units {wrong}; expected {dict(units)}")
    frame = pd.DataFrame({c: hourly[c] for c in units}, dtype="float64")
    times = pd.to_datetime(pd.Series(hourly["time"]), format="%Y-%m-%dT%H:%M", errors="raise")
    frame.insert(0, "valid_time", times.dt.tz_localize("UTC"))
    return frame


def parse_observations(
    payload: Mapping[str, Any],
    site_id: str,
    model: str,
    variables: Mapping[str, WeatherVariableConfig],
) -> pd.DataFrame:
    """Reanalysis response → rows ``(model, site_id, valid_time, *variables)``."""
    frame = hourly_frame(payload, {v: c.unit for v, c in variables.items()})
    frame.insert(0, "site_id", site_id)
    frame.insert(0, "model", model)
    return frame


def parse_forecasts(
    payload: Mapping[str, Any],
    site_id: str,
    model: str,
    variables: Mapping[str, WeatherVariableConfig],
    offsets_days: Sequence[int],
    run_every: pd.Timedelta,
    publication_delay: pd.Timedelta,
) -> pd.DataFrame:
    """Previous Runs response → one row per (offset, valid hour), with run timing."""
    units = {forecast_column(v, k): c.unit for k in offsets_days for v, c in variables.items()}
    wide = hourly_frame(payload, units)
    parts = []
    for k in offsets_days:
        part = wide[["valid_time", *(forecast_column(v, k) for v in variables)]].rename(
            columns={forecast_column(v, k): v for v in variables}
        )
        part.insert(0, "offset_days", k)
        part["run_init"] = run_init_for(part["valid_time"], k, run_every)
        part["available_at"] = available_at_for(part["run_init"], publication_delay)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["offset_days"] = frame["offset_days"].astype("int16")
    frame.insert(0, "site_id", site_id)
    frame.insert(0, "model", model)
    return frame


def clip_artefacts(
    frame: pd.DataFrame, variables: Mapping[str, WeatherVariableConfig], tolerance: float
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clip values just outside a bounded variable's physical range back onto it.

    The GFS archive stores some cloud cover values as exactly -1 % or 101 %; these are
    rounding artefacts, not weather. Only values within ``tolerance`` of the range are
    clipped (and counted); values further out are left for schema validation to reject.
    """
    out = frame.copy()
    counts: dict[str, int] = {}
    for var, spec in variables.items():
        if spec.clip_to is None:
            continue
        lo, hi = spec.clip_to
        values = out[var]
        fixable = (values.between(lo - tolerance, lo, inclusive="left")) | (
            values.between(hi, hi + tolerance, inclusive="right")
        )
        if fixable.any():
            out.loc[fixable, var] = values[fixable].clip(lo, hi)
            counts[var] = int(fixable.sum())
    return out, counts


def _keep(frame: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Rows inside ``[first, last]`` that carry at least one value (all-missing = a gap)."""
    in_range = frame["valid_time"].between(first, last)
    has_value = frame[list(WEATHER_VARIABLES)].notna().any(axis=1)
    return frame[in_range & has_value].reset_index(drop=True)


class WeatherIngestor:
    """Builds requests, parses and validates responses, and loads one month at a time."""

    def __init__(self, engine: Engine, client: OpenMeteoClient, cfg: AppConfig) -> None:
        self.engine = engine
        self.client = client
        self.cfg = cfg
        w = cfg.weather
        self.site_ids = list(w.sites)
        self.run_every = pd.Timedelta(w.forecasts.run_every)
        self.delay = pd.Timedelta(w.forecasts.publication_delay)
        self.schemas = {
            "forecasts": weather_forecasts_schema(
                self.site_ids, w.forecasts.offsets_days, self.run_every, self.delay
            ),
            "observations": weather_observations_schema(self.site_ids),
        }

    def request(self, kind: Kind, site_id: str, period: dt.date) -> OpenMeteoRequest:
        w = self.cfg.weather
        site = w.sites[site_id]
        first, last = month_hours(period)
        if kind == "forecasts":
            api: Literal["previous_runs", "archive"] = "previous_runs"
            model = w.forecasts.model
            hourly = tuple(
                forecast_column(v, k) for k in w.forecasts.offsets_days for v in w.variables
            )
        else:
            api, model, hourly = "archive", w.observations.model, tuple(w.variables)
        return OpenMeteoRequest(
            api=api,
            model=model,
            site_id=site_id,
            latitude=site.latitude,
            longitude=site.longitude,
            hourly=hourly,
            start_date=first.date(),
            end_date=last.date(),
        )

    def transform(
        self, kind: Kind, response: OpenMeteoResponse, period: dt.date
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Parse, trim to the month's hours, drop all-missing rows, clip rounding artefacts
        and validate. Returns the frame and the number of clipped values per variable."""
        w = self.cfg.weather
        req = response.request
        if kind == "forecasts":
            frame = parse_forecasts(
                response.payload,
                req.site_id,
                req.model,
                w.variables,
                w.forecasts.offsets_days,
                self.run_every,
                self.delay,
            )
        else:
            frame = parse_observations(response.payload, req.site_id, req.model, w.variables)
        frame, clipped = clip_artefacts(
            _keep(frame, *month_hours(period)), w.variables, w.clip_tolerance
        )
        try:
            return self.schemas[kind].validate(frame, lazy=True), clipped
        except pe.SchemaErrors as exc:
            raise ValueError(f"schema validation failed: {summarise_schema_errors(exc)}") from exc

    def ingest_month(
        self, kind: Kind, period: dt.date, refresh: bool = False
    ) -> WeatherMonthResult:
        """Fetch every site for one month and dataset. Failures are recorded, not raised."""
        dataset = DATASETS[kind]
        with self.engine.begin() as conn:
            ensure_regions(conn, self.cfg)
            ensure_weather_sites(conn, self.cfg)
            run_id = start_ingestion_run(conn, SOURCE, dataset, period)
        bound = log.bind(period=f"{period:%Y-%m}", dataset=dataset, run_id=run_id)
        records: list[dict[str, Any]] = []
        try:
            frames = []
            for site_id in self.site_ids:
                response = self.client.fetch(self.request(kind, site_id, period), refresh)
                records.append(response.record())
                site_frame, clipped = self.transform(kind, response, period)
                records[-1]["clipped_values"] = clipped
                frames.append(site_frame)
            frame = pd.concat(frames, ignore_index=True)
            upsert = (
                upsert_weather_forecasts if kind == "forecasts" else upsert_weather_observations
            )
            with self.engine.begin() as conn:
                loaded = upsert(conn, frame, run_id)
                finish_ingestion_run(
                    conn,
                    run_id,
                    status="success",
                    files=records,
                    rows_parsed=len(frame),
                    duplicates_dropped=0,
                    rows_loaded=loaded,
                )
            bound.info("weather.month_ok", rows=len(frame), rows_loaded=loaded)
            return WeatherMonthResult(period, dataset, "success", len(frame), loaded)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            bound.error("weather.month_failed", error=message)
            with self.engine.begin() as conn:
                finish_ingestion_run(
                    conn,
                    run_id,
                    status="failed",
                    files=records,
                    error=message[:4000],
                )
            return WeatherMonthResult(period, dataset, "failed", error=message)


def ingest_weather_range(
    engine: Engine,
    cfg: AppConfig,
    start: dt.date,
    end: dt.date,
    kinds: Sequence[Kind] = ("forecasts", "observations"),
    client: OpenMeteoClient | None = None,
    refresh: bool = False,
) -> list[WeatherMonthResult]:
    """Ingest every month in ``[start, end]`` for each dataset; continues past failures."""
    own_client = client is None
    om = client or OpenMeteoClient(cfg.weather.open_meteo, cfg.data_path(cfg.paths.raw_dir))
    try:
        ingestor = WeatherIngestor(engine, om, cfg)
        return [
            ingestor.ingest_month(kind, month, refresh)
            for kind in kinds
            for month in iter_months(start, end)
        ]
    finally:
        if own_client:
            om.close()
