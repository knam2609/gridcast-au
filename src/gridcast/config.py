"""Configuration: typed YAML config + environment-variable settings.

Non-secret, versioned settings live in ``configs/*.yaml``. Connection strings and other
environment-specific values come from environment variables (or a local ``.env``).
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-driven settings (prefix ``GRIDCAST_``)."""

    model_config = SettingsConfigDict(
        env_prefix="GRIDCAST_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    database_url: str = Field(
        default="postgresql+psycopg://gridcast:gridcast@localhost:5433/gridcast",
        description="SQLAlchemy URL of the application database.",
    )
    test_database_url: str | None = None
    config: Path = Path("configs/base.yaml")
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"


class PathsConfig(BaseModel):
    raw_dir: Path
    interim_dir: Path
    backtest_dir: Path
    reports_dir: Path


class NemConfig(BaseModel):
    market_utc_offset_hours: int = 10
    dispatch_interval_minutes: int = 5
    market_limits_file: Path

    @property
    def interval(self) -> dt.timedelta:
        return dt.timedelta(minutes=self.dispatch_interval_minutes)


class RegionConfig(BaseModel):
    name: str
    state: str
    local_tz: str
    holiday_subdiv: str
    demand_plausible_mw: tuple[float, float]


class MmsdmSourceConfig(BaseModel):
    base_url: str
    user_agent: str
    request_delay_seconds: float = 1.0
    timeout_seconds: float = 180.0
    max_retries: int = 4
    tables: list[str]
    stale_after_days: int = 75


class SourcesConfig(BaseModel):
    mmsdm: MmsdmSourceConfig


class IngestionConfig(BaseModel):
    start_month: str
    end_month: str


class ForecastConfig(BaseModel):
    issue_every_minutes: int = 30
    horizon_steps: int = 288
    availability_latency_steps: int = 0


class BacktestConfig(BaseModel):
    eval_start: dt.date
    holdout_start: dt.date
    reference_model: dict[str, str]


class BaselinesConfig(BaseModel):
    median_profile_lookback_days: int = 28


class EvaluationConfig(BaseModel):
    horizons_reported: list[int]
    peak_hours_local: tuple[int, int]
    price_spike_threshold: float


class AppConfig(BaseModel):
    """Typed view of ``configs/base.yaml``."""

    project: str
    paths: PathsConfig
    nem: NemConfig
    regions: dict[str, RegionConfig]
    sources: SourcesConfig
    ingestion: IngestionConfig
    forecast: ForecastConfig
    backtest: BacktestConfig
    baselines: BaselinesConfig
    evaluation: EvaluationConfig

    def resolve(self, path: Path) -> Path:
        """Resolve a config-relative path against the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def region(self, region_id: str) -> RegionConfig:
        try:
            return self.regions[region_id]
        except KeyError as exc:
            raise KeyError(
                f"Region {region_id!r} is not configured; known: {sorted(self.regions)}"
            ) from exc


class PriceCapEntry(BaseModel):
    value: float
    source: str


class MarketLimits(BaseModel):
    """Market floor / per-financial-year market price cap (data-quality bounds)."""

    market_floor: float
    market_price_cap: dict[str, PriceCapEntry]

    @field_validator("market_price_cap")
    @classmethod
    def _check_fy_keys(cls, v: dict[str, PriceCapEntry]) -> dict[str, PriceCapEntry]:
        for key in v:
            start, end = key.split("-")
            if len(start) != 4 or len(end) != 2 or int(end) != (int(start) + 1) % 100:
                raise ValueError(f"Financial-year key must look like '2024-25', got {key!r}")
        return v

    def cap_for(self, day: dt.date) -> float | None:
        """Market price cap for the financial year containing ``day`` (None if unknown)."""
        start_year = day.year if day.month >= 7 else day.year - 1
        entry = self.market_price_cap.get(f"{start_year}-{(start_year + 1) % 100:02d}")
        return entry.value if entry else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or get_settings().config
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open() as fh:
        return AppConfig.model_validate(yaml.safe_load(fh))


def load_market_limits(cfg: AppConfig) -> MarketLimits:
    with cfg.resolve(cfg.nem.market_limits_file).open() as fh:
        return MarketLimits.model_validate(yaml.safe_load(fh))
