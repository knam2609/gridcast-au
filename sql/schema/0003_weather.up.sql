-- 0003_weather: hourly weather for population-weighted regional features (milestone 2).
--
--   weather_sites         one row per weather point, with its population weight in a region
--   weather_forecasts     variant C: archived NWP forecasts (Open-Meteo Previous Runs), one row
--                         per (model, site, offset, valid hour), with the run that produced it
--                         and the time it became available
--   weather_observations  variant D: reanalysis "actual" weather (ERA5). Published days late,
--                         so it is a perfect-foresight oracle, never a deployable input
--
-- Conventions
--   * valid_time is the hour (UTC) a value refers to. Instant variables are values at that
--     hour; shortwave_radiation is the mean over the hour ENDING at valid_time.
--   * Rows where every variable is missing are not stored: a missing row is a gap.
--   * Day-based offsets use fixed 24-hour steps (interval '1 day' would follow DST).

CREATE TABLE weather_sites (
    site_id           text PRIMARY KEY,                      -- e.g. 'melbourne'
    region_id         text NOT NULL REFERENCES regions (region_id),
    name              text NOT NULL,
    latitude          double precision NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude         double precision NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    population        integer NOT NULL CHECK (population > 0),
    weight            double precision NOT NULL CHECK (weight > 0 AND weight <= 1),
    population_source text NOT NULL,
    updated_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX weather_sites_region_idx ON weather_sites (region_id);

CREATE TABLE weather_forecasts (
    model                text NOT NULL,                      -- Open-Meteo model id, e.g. 'gfs_seamless'
    site_id              text NOT NULL REFERENCES weather_sites (site_id),
    offset_days          smallint NOT NULL CHECK (offset_days >= 1),  -- previous_day{K}
    valid_time           timestamptz NOT NULL,
    run_init             timestamptz NOT NULL,               -- initialisation time of the source run
    available_at         timestamptz NOT NULL,               -- run_init + publication delay
    temperature_2m       double precision,                   -- °C
    apparent_temperature double precision,                   -- °C
    relative_humidity_2m double precision CHECK (relative_humidity_2m BETWEEN 0 AND 100),  -- %
    cloud_cover          double precision CHECK (cloud_cover BETWEEN 0 AND 100),           -- %
    shortwave_radiation  double precision CHECK (shortwave_radiation >= 0),  -- W/m², preceding-hour mean
    wind_speed_10m       double precision CHECK (wind_speed_10m >= 0),       -- m/s
    ingestion_run_id     bigint REFERENCES ingestion_runs (run_id),
    loaded_at            timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, site_id, offset_days, valid_time),
    CONSTRAINT weather_forecasts_hour_chk
        CHECK (date_bin('1 hour', valid_time, TIMESTAMPTZ '2000-01-01 00:00+00') = valid_time),
    -- The run must start at least offset_days x 24 h before the valid hour, and be published
    -- after it starts. The exact rule (floor to the run cadence, configured delay) is
    -- validated on load and by the weather_run_timing data-quality check.
    CONSTRAINT weather_forecasts_run_chk
        CHECK (run_init <= valid_time - make_interval(hours => 24 * offset_days)),
    CONSTRAINT weather_forecasts_available_chk CHECK (available_at > run_init),
    CONSTRAINT weather_forecasts_nonempty_chk CHECK (
        num_nonnulls(temperature_2m, apparent_temperature, relative_humidity_2m,
                     cloud_cover, shortwave_radiation, wind_speed_10m) > 0
    )
);

CREATE TABLE weather_observations (
    model                text NOT NULL,                      -- e.g. 'era5'
    site_id              text NOT NULL REFERENCES weather_sites (site_id),
    valid_time           timestamptz NOT NULL,
    temperature_2m       double precision,
    apparent_temperature double precision,
    relative_humidity_2m double precision CHECK (relative_humidity_2m BETWEEN 0 AND 100),
    cloud_cover          double precision CHECK (cloud_cover BETWEEN 0 AND 100),
    shortwave_radiation  double precision CHECK (shortwave_radiation >= 0),
    wind_speed_10m       double precision CHECK (wind_speed_10m >= 0),
    ingestion_run_id     bigint REFERENCES ingestion_runs (run_id),
    loaded_at            timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, site_id, valid_time),
    CONSTRAINT weather_observations_hour_chk
        CHECK (date_bin('1 hour', valid_time, TIMESTAMPTZ '2000-01-01 00:00+00') = valid_time),
    CONSTRAINT weather_observations_nonempty_chk CHECK (
        num_nonnulls(temperature_2m, apparent_temperature, relative_humidity_2m,
                     cloud_cover, shortwave_radiation, wind_speed_10m) > 0
    )
);
