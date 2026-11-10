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
from pydantic import BaseModel, Field, field_validator, model_validator
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
    data_root: Path | None = Field(
        default=None,
        description="Overrides paths.data_root, e.g. a directory on an external drive.",
    )
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"


class PathsConfig(BaseModel):
    data_root: Path = Path("data")  # raw/interim/backtest dirs are relative to this
    raw_dir: Path
    interim_dir: Path
    backtest_dir: Path
    reports_dir: Path  # relative to the project root (reports are versioned)


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


class DailyReportsConfig(BaseModel):
    """NEMWeb's daily archive of the five-minute dispatch reports (M2 step 6).

    Used only for days the monthly MMSDM archive does not have yet, after an exact match
    check against MMSDM on days present in both; requests reuse the ``mmsdm`` settings.
    """

    listing_url: str


class SourcesConfig(BaseModel):
    mmsdm: MmsdmSourceConfig
    nemweb_daily: DailyReportsConfig | None = None


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
    # First NEM-time day after the holdout's issue times (exclusive); the M2 holdout run
    # (`gridcast evaluate-holdout`) scores every issue time in [holdout_start, holdout_end).
    holdout_end: dt.date | None = None
    reference_model: dict[str, str]
    # M2 holdout freeze (docs/plans/m2_plan.md, step 6): the ML config hash and the commit
    # whose model and feature code produced it. A unit test fails if the config drifts.
    frozen_ml_config_hash: str | None = None
    frozen_ml_commit: str | None = None


class BaselinesConfig(BaseModel):
    median_profile_lookback_days: int = 28


class FeaturesConfig(BaseModel):
    """M2 feature builder (src/gridcast/features/, docs/feature_catalogue.md)."""

    degree_base_c: float = 18.0  # base of the cooling / heating degree features
    # Site whose temperature is also a feature on its own (besides the weighted mean).
    single_site: str = "melbourne"
    # Longer GFS publication delay for the robustness check (headline: the configured delay).
    delay_sensitivity_hours: float = 8.0


class LightGBMConfig(BaseModel):
    """Fixed LightGBM settings; the tuned ones come from ``ml.tuned_file``."""

    learning_rate: float = 0.05
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.9
    max_rounds: int = 5000
    early_stopping_rounds: int = 100
    num_threads: int = 0  # 0 = OpenMP default


class RidgeConfig(BaseModel):
    max_rows: int = 500_000  # seeded subsample of the training rows
    spline_knots: int = 6


TargetForm = Literal["residual", "direct", "anchored", "latest"]


class GridConfig(BaseModel):
    """Settings searched at every retrain on its early-stopping slice (m2_plan.md §2.4)."""

    target_forms: list[TargetForm]
    lightgbm_objective: list[Literal["l2", "l1"]]
    lightgbm_num_leaves: list[int]
    ridge_alpha: list[float]


class MlConfig(BaseModel):
    """Walk-forward ML evaluation (M2). Dates are NEM-time days."""

    first_issue: dt.date  # first ML issue time: every model has >= 12 months of training data
    # First C⁺/D⁺ issue time: the first quarterly retrain with >= 12 months of the GFS
    # multi-variable archive (which starts 2024-01-19).
    plus_first_issue: dt.date
    plus_train_start: dt.date  # first C⁺/D⁺ training issue day (the archive start)
    retrain_every_months: int = 3
    early_stopping_weeks: int = 8
    sample_every_steps: int = 3  # training targets every N steps, staggered across issue times
    lead_bands: list[tuple[int, int]] = [(1, 288)]  # one fit per band (inclusive horizon steps)
    seed: int = 2609
    lightgbm: LightGBMConfig = LightGBMConfig()
    ridge: RidgeConfig = RidgeConfig()
    grid: GridConfig
    model_dir: Path = Path("models")  # under the data root

    @model_validator(mode="after")
    def _ordered(self) -> MlConfig:
        if not self.plus_train_start < self.plus_first_issue:
            raise ValueError("ml.plus_train_start must precede ml.plus_first_issue")
        if self.retrain_every_months < 1 or self.sample_every_steps < 1:
            raise ValueError("retrain_every_months and sample_every_steps must be positive")
        return self


class KnownEvent(BaseModel):
    """A local day worth showing in the failure cases whatever its rank."""

    day: dt.date
    label: str


class AemoBenchmarkConfig(BaseModel):
    """AEMO's own demand forecasts, a benchmark only (never a model input); m2_plan.md §2.6."""

    ingest_start_month: str  # YYYY-MM, archive months downloaded and loaded
    ingest_end_month: str
    first_issue: dt.date  # first NEM-time issue day compared; development ends at holdout_start
    # Data-quality bounds on publication time (LASTCHANGED) relative to the nominal run time.
    predispatch_published_within_minutes: float = 15.0  # after the run time
    p5min_published_within_minutes: float = 10.0  # before the run time


class EvaluationConfig(BaseModel):
    horizons_reported: list[int]
    peak_hours_local: tuple[int, int]
    price_spike_threshold: float
    # Weather segments (M2), from ERA5 population-weighted values on the local day.
    hot_day_max_temp_c: float = 35.0
    cloudy_day_mean_cloud_pct: float = 80.0
    daytime_hours_local: tuple[int, int] = (9, 17)
    # M2 step 4 (m2_plan.md §2.5): weather-feature drop/add test at these retrains, on
    # their early-stopping slices, flagging any feature that improves MAE by more than
    # ``ablation_flag_share`` on its own.
    weather_ablation_retrains: list[dt.date] = Field(default_factory=list)
    ablation_flag_share: float = Field(0.15, gt=0, lt=1)
    # Failure cases: the worst local days at leads 2-24 h, plus known events.
    failure_days: int = Field(10, gt=0)
    known_events: list[KnownEvent] = Field(default_factory=list)


class IntervalGridConfig(BaseModel):
    """Conformal candidate streams (m3_plan.md §2.2): the product of these, in this order."""

    window_weeks: list[int]
    groupings: list[Literal["lead", "lead_tod"]]
    aci_gammas: list[float]  # 0 = ACI off


class IntervalsConfig(BaseModel):
    """Milestone 3: prediction intervals for demand (docs/plans/m3_plan.md).

    Not part of the frozen M2 hash (``backtest.frozen_ml_config_hash``).
    """

    # First scored issue day (a quarterly retrain of ml.first_issue's schedule). Development
    # runs to backtest.holdout_start; issue times before first_issue are pre-scoring data.
    first_issue: dt.date
    levels: list[int]  # central levels, percent
    headline_level: int
    # Stored development point forecasts (M2 run 2, evaluation set "main") and the point
    # models that get conformal intervals.
    source_batch: str
    point_models: list[str]
    selection_weeks: int = 8  # selection slice before each retrain (like early stopping)
    grid: IntervalGridConfig
    tod_block_starts: list[int]  # local hours starting the time-of-day blocks
    # Extreme-event groups known at the issue time: forecast (GFS, population-weighted)
    # daily maximum of the target's local day at least / at most these.
    hot_day_max_temp_c: float = 35.0
    cold_day_max_temp_c: float
    output_dir: Path = Path("intervals")  # under the data root

    @model_validator(mode="after")
    def _consistent(self) -> IntervalsConfig:
        if self.headline_level not in self.levels:
            raise ValueError("intervals.headline_level must be one of intervals.levels")
        if any(not 0 < lv < 100 for lv in self.levels) or self.levels != sorted(set(self.levels)):
            raise ValueError("intervals.levels must be ascending percentages in (0, 100)")
        return self


# Hourly weather variables, in storage order. They are columns of weather_forecasts and
# weather_observations (migration 0003), so the config must list exactly these.
WEATHER_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
)


class WeatherVariableConfig(BaseModel):
    unit: str  # unit string Open-Meteo must report (checked on every response)
    kind: Literal["instant", "preceding_hour_mean"]
    plausible: tuple[float, float]
    # Physical range of a bounded variable (e.g. 0-100 %). Values outside it by no more than
    # weather.clip_tolerance are archive rounding artefacts: clipped on load and counted.
    clip_to: tuple[float, float] | None = None


class WeatherSiteConfig(BaseModel):
    name: str
    region_id: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    sua_population: dict[str, int] = Field(min_length=1)  # ABS SUA name -> ERP

    @property
    def population(self) -> int:
        return sum(self.sua_population.values())


class PopulationSourceConfig(BaseModel):
    source: str
    year: int


class ExpectedMissingConfig(BaseModel):
    """A range of valid hours (UTC, inclusive) where an archived value is known to be missing."""

    reason: str
    variables: list[str] = Field(min_length=1)
    offset_days: int
    first: dt.datetime | None  # None = from the start of the data
    last: dt.datetime

    @field_validator("first", "last")
    @classmethod
    def _tz_aware(cls, v: dt.datetime | None) -> dt.datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("expected_missing times must include a UTC offset")
        return v


class WeatherForecastConfig(BaseModel):
    model: str
    offsets_days: list[int] = Field(min_length=1)
    run_every_hours: int = Field(gt=0)
    publication_delay_hours: float = Field(gt=0)
    expected_missing: list[ExpectedMissingConfig] = []

    @property
    def run_every(self) -> dt.timedelta:
        return dt.timedelta(hours=self.run_every_hours)

    @property
    def publication_delay(self) -> dt.timedelta:
        return dt.timedelta(hours=self.publication_delay_hours)


class WeatherObservationConfig(BaseModel):
    model: str


class OpenMeteoConfig(BaseModel):
    previous_runs_url: str
    archive_url: str
    user_agent: str
    request_delay_seconds: float = 1.0
    timeout_seconds: float = 120.0
    max_retries: int = 6
    retry_backoff_seconds: float = 5.0  # exponential backoff base for 429/5xx/transport errors
    settle_days: int = 10


class WeatherConfig(BaseModel):
    population: PopulationSourceConfig
    sites: dict[str, WeatherSiteConfig] = Field(min_length=1)
    variables: dict[str, WeatherVariableConfig]
    clip_tolerance: float = Field(default=0.0, ge=0)
    forecasts: WeatherForecastConfig
    observations: WeatherObservationConfig
    open_meteo: OpenMeteoConfig

    @model_validator(mode="after")
    def _consistent(self) -> WeatherConfig:
        if tuple(self.variables) != WEATHER_VARIABLES:
            raise ValueError(f"weather.variables must be exactly {list(WEATHER_VARIABLES)}")
        offsets = set(self.forecasts.offsets_days)
        if min(offsets) < 1:
            raise ValueError("forecast offsets must be >= 1 day (day 0 is never available)")
        for gap in self.forecasts.expected_missing:
            unknown = set(gap.variables) - set(WEATHER_VARIABLES)
            if unknown or gap.offset_days not in offsets:
                raise ValueError(f"expected_missing entry {gap.reason!r} has unknown keys")
        return self

    def sites_for(self, region_id: str) -> dict[str, WeatherSiteConfig]:
        return {k: s for k, s in self.sites.items() if s.region_id == region_id}

    def site_weights(self, region_id: str) -> dict[str, float]:
        """Population share of each of the region's sites (sums to 1)."""
        sites = self.sites_for(region_id)
        if not sites:
            raise KeyError(f"no weather sites configured for region {region_id!r}")
        total = sum(s.population for s in sites.values())
        return {k: s.population / total for k, s in sites.items()}


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
    weather: WeatherConfig
    features: FeaturesConfig
    ml: MlConfig
    aemo: AemoBenchmarkConfig
    intervals: IntervalsConfig

    @model_validator(mode="after")
    def _weather_regions_known(self) -> AppConfig:
        unknown = {s.region_id for s in self.weather.sites.values()} - set(self.regions)
        if unknown:
            raise ValueError(f"weather sites reference unknown regions {sorted(unknown)}")
        if self.features.single_site not in self.weather.sites:
            raise ValueError(f"features.single_site {self.features.single_site!r} is not a site")
        if not self.ml.first_issue <= self.ml.plus_first_issue < self.backtest.holdout_start:
            raise ValueError("expected ml.first_issue <= ml.plus_first_issue < holdout_start")
        end = self.backtest.holdout_end
        if end is not None and end <= self.backtest.holdout_start:
            raise ValueError("expected backtest.holdout_start < backtest.holdout_end")
        expected = 1
        for lo, hi in self.ml.lead_bands:
            if lo != expected or hi < lo:
                raise ValueError("ml.lead_bands must be contiguous, ordered and start at 1")
            expected = hi + 1
        if expected != self.forecast.horizon_steps + 1:
            raise ValueError("ml.lead_bands must end at forecast.horizon_steps")
        if not self.ml.first_issue < self.intervals.first_issue < self.backtest.holdout_start:
            raise ValueError("expected ml.first_issue < intervals.first_issue < holdout_start")
        return self

    def resolve(self, path: Path) -> Path:
        """Resolve a config-relative path against the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def data_root(self) -> Path:
        """Absolute data root. Created if it lives inside the project; otherwise it must
        already exist (an unmounted external drive fails loudly instead of silently
        writing somewhere else)."""
        root = self.resolve(self.paths.data_root)
        if not root.exists():
            if root.is_relative_to(PROJECT_ROOT):
                root.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Data root {root} does not exist. If it is on an external drive, "
                    "mount the drive (or unset GRIDCAST_DATA_ROOT to use ./data)."
                )
        return root

    def data_path(self, path: Path) -> Path:
        """Resolve a data path (raw/interim/backtests) against the data root."""
        return path if path.is_absolute() else self.data_root() / path

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
        cfg = AppConfig.model_validate(yaml.safe_load(fh))
    override = get_settings().data_root
    if override is not None:
        cfg.paths.data_root = override.expanduser()
    return cfg


def load_market_limits(cfg: AppConfig) -> MarketLimits:
    with cfg.resolve(cfg.nem.market_limits_file).open() as fh:
        return MarketLimits.model_validate(yaml.safe_load(fh))
