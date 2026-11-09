"""Feature catalogue: the single source of feature names, definitions and leakage reasoning.

The builder (:mod:`gridcast.features.builder`) takes its column names and order from
here, and ``gridcast feature-catalogue`` renders ``docs/feature_catalogue.md`` from it.
Unit tests fail if the builder's columns and the catalogue disagree, or if the rendered
document is stale.
"""

from __future__ import annotations

from dataclasses import dataclass

from gridcast.config import AppConfig


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    definition: str
    oldest: str  # earliest data the feature reads, relative to o (issue) or t (target)


@dataclass(frozen=True, slots=True)
class GroupSpec:
    key: str
    title: str
    source: str
    known: str  # when the inputs are known
    safety: str  # why the group cannot leak information from after o
    features: tuple[FeatureSpec, ...]


@dataclass(frozen=True, slots=True)
class VariantSpec:
    id: str
    label: str
    weather: str  # "none", "forecast" or "observed"
    groups: tuple[str, ...]
    summary: str

    @property
    def oracle(self) -> bool:
        return self.weather == "observed"


DEMAND = "`market_intervals.total_demand_mw` (AEMO `DISPATCHREGIONSUM.TOTALDEMAND`)"

GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        key="lead",
        title="Lead time",
        source="Forecast design",
        known="By construction",
        safety="Depends only on h.",
        features=(FeatureSpec("lead_minutes", "5 × h, for h = 1…288", "none"),),
    ),
    GroupSpec(
        key="calendar",
        title="Calendar at the target",
        source="`calendar_days` (Victorian public holidays, `holidays` package) and the "
        "local clock (Australia/Melbourne) at the target interval's start",
        known="In advance, for any date in `calendar_days`",
        safety="Calendar attributes are fixed long before t. Building the context fails if a "
        "target date is missing from `calendar_days`, so a stale calendar cannot "
        "silently mark a holiday as a working day.",
        features=(
            FeatureSpec("cal_slot", "Local 5-minute slot of the day (0–287)", "calendar"),
            FeatureSpec("cal_day_of_week", "Local day of the week (0 = Monday)", "calendar"),
            FeatureSpec("cal_month", "Local month (1–12)", "calendar"),
            FeatureSpec("cal_non_working", "1 on weekends and public holidays", "calendar"),
            FeatureSpec("cal_public_holiday", "1 on Victorian public holidays", "calendar"),
            FeatureSpec(
                "cal_dst_change_day", "1 on local days of 23 or 25 hours (clock change)", "calendar"
            ),
            FeatureSpec("cal_tod_sin", "sin(2π × local minute of day / 1440)", "calendar"),
            FeatureSpec("cal_tod_cos", "cos(2π × local minute of day / 1440)", "calendar"),
        ),
    ),
    GroupSpec(
        key="calendar_yesterday",
        title="Calendar 24 hours before the target",
        source="As above, for the interval ending at t − 24 h (the interval that supplies "
        "`dem_same_time_yesterday`)",
        known="In advance",
        safety="Calendar attributes only.",
        features=(
            FeatureSpec(
                "cal_yesterday_non_working",
                "`cal_non_working` of the interval at t − 24 h",
                "calendar",
            ),
            FeatureSpec(
                "cal_yesterday_public_holiday",
                "`cal_public_holiday` of the interval at t − 24 h",
                "calendar",
            ),
        ),
    ),
    GroupSpec(
        key="demand_now",
        title="Demand at the issue time",
        source=DEMAND,
        known="Intervals ending at or before the data cutoff c = o − latency",
        safety="Every window ends at c. Values are computed from the intervals that end in "
        "the window and nothing else, so later intervals cannot change them. Any "
        "missing interval in a window makes the feature missing.",
        features=(
            FeatureSpec("dem_now", "y(c)", "c"),
            FeatureSpec("dem_now_vs_yesterday", "y(c) − y(c − 24 h)", "c − 24 h"),
            FeatureSpec("dem_change_1h", "y(c) − y(c − 1 h)", "c − 1 h"),
            FeatureSpec(
                "dem_mean_1h", "Mean of the 12 intervals ending in (c − 1 h, c]", "c − 55 min"
            ),
            FeatureSpec("dem_std_1h", "Standard deviation (ddof = 0) of the same 12", "c − 55 min"),
            FeatureSpec(
                "dem_mean_24h",
                "Mean of the 288 intervals ending in (c − 24 h, c]",
                "c − 23 h 55 min",
            ),
            FeatureSpec(
                "dem_std_24h", "Standard deviation (ddof = 0) of the same 288", "c − 23 h 55 min"
            ),
        ),
    ),
    GroupSpec(
        key="demand_seasonal",
        title="Demand at the same time yesterday and last week",
        source=DEMAND,
        known="t − k × period ≤ c, with the smallest such k ≥ 1 (with latency 0 and h ≤ 288, "
        "k = 1: exactly 24 h and 7 days before t)",
        safety="The source position is checked against c for every row (`assert_known`); "
        "with h ≤ 288 and latency 0, t − 24 h ≤ o always holds. "
        "`dem_same_time_yesterday` is also the daily-naive forecast, the base of the "
        "residual target.",
        features=(
            FeatureSpec("dem_same_time_yesterday", "y(t − 24 h)", "o − 23 h 55 min (h = 1)"),
            FeatureSpec("dem_same_time_last_week", "y(t − 7 d)", "o − 6 d 23 h 55 min (h = 1)"),
            FeatureSpec("dem_yesterday_vs_last_week", "y(t − 24 h) − y(t − 7 d)", "as above"),
        ),
    ),
    GroupSpec(
        key="weather",
        title="Weather at the target",
        source="C: GFS forecasts (`weather_forecasts`, Open-Meteo Previous Runs). "
        "D: ERA5 reanalysis (`weather_observations`). Weighted over five sites by population",
        known="C: the freshest forecast run with `available_at` ≤ o. D: never at "
        "forecast time (ERA5 is published about 5 days late)",
        safety="C reads a 5-minute value only if both hours it interpolates between have "
        "`available_at` ≤ o (see *Forecast weather selection*). D is a perfect-foresight "
        "oracle by construction and is exempt from the weather leakage test.",
        features=(
            FeatureSpec("wx_temp", "Population-weighted 2 m temperature (°C)", "valid at t"),
            FeatureSpec("wx_cooling_degrees", "max(`wx_temp` − {base}, 0)", "valid at t"),
            FeatureSpec("wx_heating_degrees", "max({base} − `wx_temp`, 0)", "valid at t"),
            FeatureSpec("wx_temp_sq", "`wx_temp`²", "valid at t"),
            FeatureSpec("wx_temp_single_site", "2 m temperature at {site} (°C)", "valid at t"),
        ),
    ),
    GroupSpec(
        key="weather_plus",
        title="Extra weather at the target",
        source="As for *Weather at the target*. The GFS archive of these variables starts "
        "2024-01-19 12:00 UTC (day 1) and 2024-01-20 12:00 UTC (day 2)",
        known="As above",
        safety="As above.",
        features=(
            FeatureSpec("wx_apparent_temp", "Weighted apparent temperature (°C)", "valid at t"),
            FeatureSpec("wx_humidity", "Weighted relative humidity at 2 m (%)", "valid at t"),
            FeatureSpec("wx_cloud_cover", "Weighted total cloud cover (%)", "valid at t"),
            FeatureSpec(
                "wx_shortwave",
                "Weighted shortwave radiation (W/m²), mean over the hour ending ceil₁ₕ(t)",
                "hour ending ceil₁ₕ(t)",
            ),
        ),
    ),
)

BASE_GROUPS = ("lead", "calendar", "calendar_yesterday", "demand_now", "demand_seasonal")
VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("A", "A", "none", BASE_GROUPS, "No weather"),
    VariantSpec(
        "C",
        "C",
        "forecast",
        (*BASE_GROUPS, "weather"),
        "Archived GFS temperature forecasts (headline)",
    ),
    VariantSpec(
        "D",
        "D",
        "observed",
        (*BASE_GROUPS, "weather"),
        "ERA5 temperature: perfect-foresight mirror of C",
    ),
    VariantSpec(
        "Cplus",
        "C⁺",
        "forecast",
        (*BASE_GROUPS, "weather", "weather_plus"),
        "All archived GFS variables (secondary)",
    ),
    VariantSpec(
        "Dplus",
        "D⁺",
        "observed",
        (*BASE_GROUPS, "weather", "weather_plus"),
        "ERA5 mirror of C⁺",
    ),
)

# Weather source series (gridcast.features.weather.SOURCES) each weather group needs.
GROUP_SOURCES: dict[str, tuple[str, ...]] = {
    "weather": ("temp_weighted", "temp_single"),
    "weather_plus": (
        "apparent_temp_weighted",
        "humidity_weighted",
        "cloud_weighted",
        "radiation_weighted",
    ),
}

_GROUPS_BY_KEY = {g.key: g for g in GROUPS}
_VARIANTS_BY_ID = {v.id: v for v in VARIANTS}


def variant(variant_id: str) -> VariantSpec:
    try:
        return _VARIANTS_BY_ID[variant_id]
    except KeyError as exc:
        raise KeyError(f"unknown variant {variant_id!r}; known: {list(_VARIANTS_BY_ID)}") from exc


def group(key: str) -> GroupSpec:
    return _GROUPS_BY_KEY[key]


def feature_names(variant_id: str) -> list[str]:
    """Column names of a variant's feature matrix, in order."""
    return [f.name for key in variant(variant_id).groups for f in group(key).features]


def required_sources(variant_id: str) -> list[str]:
    return [s for key in variant(variant_id).groups for s in GROUP_SOURCES.get(key, ())]


# --------------------------------------------------------------------------- rendering


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" :--- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_catalogue(cfg: AppConfig) -> str:
    """Markdown for ``docs/feature_catalogue.md`` (deterministic: no timestamps)."""
    base = f"{cfg.features.degree_base_c:g} °C"
    site = cfg.weather.sites[cfg.features.single_site].name
    fc = cfg.weather.forecasts
    first = {
        "A": cfg.ml.first_issue,
        "C": cfg.ml.first_issue,
        "D": cfg.ml.first_issue,
        "Cplus": cfg.ml.plus_first_issue,
        "Dplus": cfg.ml.plus_first_issue,
    }
    weather_label = {
        "none": "none",
        "forecast": f"GFS forecast (`{fc.model}`)",
        "observed": f"ERA5 (`{cfg.weather.observations.model}`)",
    }
    out: list[str] = [
        "# Feature catalogue (M2)",
        "",
        "> **Generated** by `gridcast feature-catalogue` from "
        "`src/gridcast/features/catalogue.py`. Do not edit by hand. A unit test fails if "
        "this file is out of date. Measured coverage and missing rates are in the "
        "generated [`reports/m2_feature_audit.md`](../reports/m2_feature_audit.md).",
        "",
        "## Notation",
        "",
        "- *o*: issue time (every 30 minutes). *h*: horizon, 1…288 five-minute steps. "
        "*t* = *o* + 5 min × *h*: the target interval's end.",
        "- *c* = *o* − latency: the data cutoff. Intervals ending at or before *c* are known "
        f"at *o*. The configured latency is {cfg.forecast.availability_latency_steps} "
        "intervals, so *c* = *o*.",
        "- y(·): Victorian total demand (MW) of the interval ending at that time.",
        "- Times are UTC on a complete 5-minute grid. Local (Australia/Melbourne) time is "
        "used only for calendar features.",
        "",
        "## Variants",
        "",
        "C and D have identical feature names, and so do C⁺ and D⁺: only the weather "
        "source differs, so a C-vs-D gap measures the cost of weather-forecast error.",
        "",
        _table(
            ["Variant", "Weather", "Feature groups", "Features", "First issue (NEM date)", "Use"],
            [
                [
                    v.label,
                    weather_label[v.weather],
                    ", ".join(group(k).title for k in v.groups[len(BASE_GROUPS) :]) or "—",
                    str(len(feature_names(v.id))),
                    str(first[v.id]),
                    v.summary,
                ]
                for v in VARIANTS
            ],
        ),
        "",
        "Every variant also has the base groups: lead time, calendar (target and 24 h "
        "earlier) and demand (issue time and seasonal lags).",
        "",
        "## Features",
        "",
    ]
    for g in GROUPS:
        users = [v.label for v in VARIANTS if g.key in v.groups]
        out += [
            f"### {g.title}",
            "",
            f"- **Source:** {g.source}.",
            f"- **Known:** {g.known}.",
            f"- **Why it is safe:** {g.safety}",
            f"- **Variants:** {', '.join(users)}.",
            "",
            _table(
                ["Feature", "Definition", "Oldest data read"],
                [
                    [f"`{f.name}`", f.definition.format(base=base, site=site), f.oldest]
                    for f in g.features
                ],
            ),
            "",
        ]
    delay = fc.publication_delay_hours
    out += [
        "## Forecast weather selection (C and C⁺)",
        "",
        f"1. Each archived GFS value (`{fc.model}`, offsets "
        f"{', '.join(f'day {k}' for k in fc.offsets_days)}) carries `run_init` = "
        f"floor_{fc.run_every_hours}h(valid time − 24 h × offset) and "
        f"`available_at` = `run_init` + {delay:g} h (verified in step 0, "
        "`docs/m2_data_spike.md` §3).",
        "2. Hourly values are weighted over the sites, then interpolated to 5 minutes. A "
        "5-minute value exists only if both hours it uses exist, and it is available from "
        "the **later** hour's `available_at`.",
        "3. For each pair (*o*, *t*), the builder takes the **freshest offset** whose every "
        "required value at *t* exists and has `available_at` ≤ *o*. All weather features "
        "of a pair come from that one offset. If no offset qualifies, the pair has no C "
        "weather and is excluded for every model and baseline (the GFS gap rule, "
        "m2_plan.md §2.4).",
        "4. A sensitivity run can recompute `available_at` with a longer delay "
        "(`publication_delay` override). The headline uses the configured delay.",
        "",
        "## Target",
        "",
        "The models predict the correction to the same time yesterday, "
        "r = y(*t*) − `dem_same_time_yesterday`, and forecast "
        "`dem_same_time_yesterday` + r̂. Step 3 also tries predicting y(*t*) directly as "
        "an ablation on development data.",
        "",
        "## Leakage audit",
        "",
        "Property tests (`tests/unit/test_feature_leakage.py`) rebuild the features after "
        "perturbing data that is unknown at *o* and require bit-for-bit identical results:",
        "",
        "- **Future demand:** every value after *c* is replaced by noise. All five variants, "
        "including D and D⁺, must be unchanged.",
        "- **Unpublished forecasts:** every hourly forecast row with `available_at` > *o* is "
        "replaced by noise, or deleted. C and C⁺ must be unchanged. This covers the later "
        "hour an interpolated value uses.",
        "- **Controls:** perturbing forecast rows that *are* available changes C, and "
        "perturbing future ERA5 changes D. These show that the tests can fail.",
        "- **Exempt:** D and D⁺ under weather perturbation. They are the labelled "
        "perfect-foresight oracles.",
        "",
        "Rule for steps 3–4: any single feature that improves pooled MAE by more than 15% "
        "on its own (added or removed one at a time) is investigated for leakage before the "
        "result is believed.",
        "",
        "## Deferred",
        "",
        "- Weather compared with yesterday's weather. It needs an analysis of yesterday's "
        "weather that is available at *o*, and ERA5 is published about 5 days late.",
        "",
    ]
    return "\n".join(out)
