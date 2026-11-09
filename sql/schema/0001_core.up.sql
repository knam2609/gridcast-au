-- 0001_core: market data, lineage, calendar, model runs, forecasts, metrics, data quality.
--
-- Conventions
--   * Every timestamp is timestamptz (stored as UTC). AEMO "NEM time" (fixed UTC+10) is
--     converted on ingestion; local Australia/Melbourne time is derived only in views.
--   * Market intervals are keyed by (region_id, interval_end), matching AEMO's
--     interval-ending SETTLEMENTDATE convention.
--   * Region-specific values (e.g. holiday jurisdiction, local time zone) live in `regions`
--     so new NEM regions are added by inserting rows, not by changing the schema.

CREATE TABLE regions (
    region_id            text PRIMARY KEY,                 -- AEMO REGIONID, e.g. 'VIC1'
    name                 text NOT NULL,
    holiday_jurisdiction text NOT NULL,                    -- e.g. 'VIC'
    local_tz             text NOT NULL,                    -- IANA zone, e.g. 'Australia/Melbourne'
    created_at           timestamptz NOT NULL DEFAULT now()
);

-- One row per (source, dataset, month) ingestion attempt: lineage + monitoring.
CREATE TABLE ingestion_runs (
    run_id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source             text NOT NULL,                      -- e.g. 'mmsdm'
    dataset            text NOT NULL,                      -- e.g. 'market_intervals'
    period_month       date NOT NULL,                      -- first day of the archive month
    files              jsonb NOT NULL DEFAULT '[]',        -- [{table, file, sha256, bytes}]
    rows_parsed        integer,
    duplicates_dropped integer,
    rows_loaded        integer,
    status             text NOT NULL DEFAULT 'running'
                       CHECK (status IN ('running', 'success', 'failed')),
    error              text,
    started_at         timestamptz NOT NULL DEFAULT now(),
    finished_at        timestamptz
);
CREATE INDEX ingestion_runs_lookup_idx
    ON ingestion_runs (source, dataset, period_month, started_at DESC);

-- 5-minute regional market outcomes.
CREATE TABLE market_intervals (
    region_id               text NOT NULL REFERENCES regions (region_id),
    interval_start          timestamptz NOT NULL,
    interval_end            timestamptz NOT NULL,
    total_demand_mw         double precision,              -- DISPATCHREGIONSUM.TOTALDEMAND
    rrp_aud_mwh             double precision,              -- DISPATCHPRICE.RRP
    net_interchange_mw      double precision,
    available_generation_mw double precision,
    apc_flag                boolean NOT NULL DEFAULT false,
    market_suspended_flag   boolean NOT NULL DEFAULT false,
    price_status            text,
    ingestion_run_id        bigint REFERENCES ingestion_runs (run_id),
    loaded_at               timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (region_id, interval_end),
    CONSTRAINT market_intervals_order_chk CHECK (interval_start < interval_end),
    -- Hard sanity bound only (physically/contractually impossible values). Per-financial-
    -- year price-cap checks are data-quality checks, which flag rather than reject.
    CONSTRAINT market_intervals_rrp_sanity_chk
        CHECK (rrp_aud_mwh IS NULL OR rrp_aud_mwh BETWEEN -1000 AND 100000)
);
CREATE UNIQUE INDEX market_intervals_start_uidx ON market_intervals (region_id, interval_start);

-- Local-calendar attributes per jurisdiction and local date (known in advance => leakage-safe).
CREATE TABLE calendar_days (
    jurisdiction      text NOT NULL,
    local_date        date NOT NULL,
    local_tz          text NOT NULL,
    is_weekend        boolean NOT NULL,
    is_public_holiday boolean NOT NULL,
    holiday_name      text,
    is_dst_change_day boolean NOT NULL,
    n_intervals       smallint NOT NULL,                   -- 5-min intervals in the local day: 276/288/300
    PRIMARY KEY (jurisdiction, local_date),
    CONSTRAINT calendar_days_n_intervals_chk CHECK (n_intervals IN (276, 288, 300))
);

-- One row per trained/evaluated model configuration (baselines included).
CREATE TABLE model_runs (
    run_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id      uuid NOT NULL,                           -- groups runs from one invocation
    model_name    text NOT NULL,
    model_kind    text NOT NULL CHECK (model_kind IN ('baseline', 'ml')),
    target        text NOT NULL CHECK (target IN ('demand', 'price')),
    region_id     text NOT NULL REFERENCES regions (region_id),
    eval_start    timestamptz,
    eval_end      timestamptz,
    holdout_start timestamptz,
    config        jsonb NOT NULL DEFAULT '{}',
    git_sha       text,
    artifact_path text,                                    -- e.g. backtest predictions parquet
    status        text NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running', 'success', 'failed')),
    created_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);
CREATE INDEX model_runs_lookup_idx ON model_runs (region_id, target, created_at DESC);

-- Aggregated evaluation metrics. horizon = 0 means "all horizons pooled".
CREATE TABLE forecast_metrics (
    run_id  bigint NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    split   text NOT NULL CHECK (split IN ('dev', 'holdout', 'all')),
    segment text NOT NULL,
    horizon smallint NOT NULL CHECK (horizon >= 0),
    metric  text NOT NULL,
    value   double precision,
    n       bigint NOT NULL,
    PRIMARY KEY (run_id, split, segment, horizon, metric)
);

-- Point (and later interval) forecasts. Full backtests live in parquet; this table holds
-- operational/latest forecasts and small backtest samples.
CREATE TABLE forecasts (
    run_id         bigint NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    region_id      text NOT NULL REFERENCES regions (region_id),
    issue_time     timestamptz NOT NULL,
    target_end     timestamptz NOT NULL,
    horizon        smallint NOT NULL CHECK (horizon >= 1),
    point          double precision NOT NULL,
    lower          double precision,
    upper          double precision,
    interval_level real CHECK (interval_level IS NULL OR (interval_level > 0 AND interval_level < 1)),
    PRIMARY KEY (run_id, issue_time, target_end),
    CONSTRAINT forecasts_future_chk CHECK (target_end > issue_time)
);

-- Results of each data-quality check execution.
CREATE TABLE dq_results (
    result_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    check_run_id uuid NOT NULL,
    check_name   text NOT NULL,
    region_id    text REFERENCES regions (region_id),
    status       text NOT NULL CHECK (status IN ('pass', 'warn', 'fail')),
    n_violations bigint NOT NULL,
    details      jsonb NOT NULL DEFAULT '{}',
    run_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX dq_results_lookup_idx ON dq_results (check_name, region_id, run_at DESC);

-- Market intervals with local-time calendar attributes (local time from interval_start).
CREATE VIEW v_market_intervals_local AS
SELECT
    m.region_id,
    m.interval_start,
    m.interval_end,
    (m.interval_start AT TIME ZONE r.local_tz)                       AS local_start,
    (m.interval_start AT TIME ZONE r.local_tz)::date                 AS local_date,
    extract(hour FROM m.interval_start AT TIME ZONE r.local_tz)::int AS local_hour,
    extract(isodow FROM m.interval_start AT TIME ZONE r.local_tz)::int AS local_isodow,
    c.is_weekend,
    c.is_public_holiday,
    c.holiday_name,
    c.is_dst_change_day,
    m.total_demand_mw,
    m.rrp_aud_mwh,
    m.net_interchange_mw,
    m.available_generation_mw,
    m.apc_flag,
    m.market_suspended_flag
FROM market_intervals m
JOIN regions r USING (region_id)
LEFT JOIN calendar_days c
       ON c.jurisdiction = r.holiday_jurisdiction
      AND c.local_date = (m.interval_start AT TIME ZONE r.local_tz)::date;
