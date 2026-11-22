-- 0007_anomalies: supply data for anomaly evidence, the anomaly event catalogue, and anomaly
-- runs in model_runs (milestone 4, docs/plans/m4_plan.md).
--
-- generation_intervals: regional supply totals from DISPATCHREGIONSUM (the pricing run,
-- INTERVENTION = 0), one row per (region, interval, kind). Evidence and the Isolation Forest
-- only; never a demand-model input.
--   kind 'wind' / 'solar'  semi-scheduled units: cleared_mw = SS_<kind>_CLEAREDMW,
--                          uigf_mw = SS_<kind>_UIGF (AEMO's unconstrained intermittent
--                          generation forecast, every table version), available_mw =
--                          SS_<kind>_AVAILABILITY (table version 8 on, May 2024; NULL before).
--                          Curtailment is measured against uigf_mw (m4_plan.md §2.3).
--   kind 'dispatchable'    cleared_mw = DISPATCHABLEGENERATION.
--   kind 'battery'         bidirectional units: cleared_mw = BDU_CLEAREDMW_GEN,
--                          load_mw = BDU_CLEAREDMW_LOAD. Rows exist only where the source
--                          has values (from 22 May 2024); batteries registered otherwise sit
--                          inside the dispatchable totals.
-- TOTALINTERMITTENTGENERATION is an allowance in AEMO's demand forecast, not an output, and
-- is not loaded. Negative values and cleared above UIGF are data-quality findings, not
-- constraints, so a month is never rejected for them.

CREATE TABLE generation_intervals (
    region_id        text NOT NULL REFERENCES regions (region_id),
    interval_end     timestamptz NOT NULL,
    kind             text NOT NULL CHECK (kind IN ('wind', 'solar', 'dispatchable', 'battery')),
    cleared_mw       double precision NOT NULL,
    uigf_mw          double precision,
    available_mw     double precision,
    load_mw          double precision,
    ingestion_run_id bigint REFERENCES ingestion_runs (run_id),
    loaded_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (region_id, interval_end, kind),
    CONSTRAINT generation_grid_chk CHECK (
        date_bin(INTERVAL '5 minutes', interval_end, TIMESTAMPTZ '2000-01-01 00:00+00') = interval_end
    ),
    CONSTRAINT generation_columns_chk CHECK (
        CASE kind
            WHEN 'wind' THEN uigf_mw IS NOT NULL AND load_mw IS NULL
            WHEN 'solar' THEN uigf_mw IS NOT NULL AND load_mw IS NULL
            WHEN 'dispatchable' THEN uigf_mw IS NULL AND available_mw IS NULL AND load_mw IS NULL
            WHEN 'battery' THEN uigf_mw IS NULL AND available_mw IS NULL AND load_mw IS NOT NULL
        END
    )
);

-- Anomaly events (m4_plan.md §2.4): alerts within 30 minutes of each other, merged. Times
-- are interval ends, as in market_intervals. One anomaly run (model_runs, kind 'anomaly',
-- target 'market') stores the events of its scored period; GET /anomalies (M5) reads here.
CREATE TABLE anomalies (
    event_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id            bigint NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    region_id         text NOT NULL REFERENCES regions (region_id),
    split             text NOT NULL CHECK (split IN ('dev', 'holdout')),
    first_interval_end timestamptz NOT NULL,
    last_interval_end  timestamptz NOT NULL,
    peak_interval_end  timestamptz NOT NULL,       -- lowest p, ties by largest score
    tier              text NOT NULL CHECK (tier IN ('notable', 'severe')),
    detectors         text[] NOT NULL,            -- every detector (lines included) that fired
    min_p             double precision,           -- lowest p of a calibrated detector; NULL if lines only
    categories        text[] NOT NULL,
    lines_only        boolean GENERATED ALWAYS AS (
                          detectors <@ ARRAY['price_line_300', 'price_line_5000']::text[]
                      ) STORED,
    evidence          jsonb NOT NULL DEFAULT '{}',
    summary           text NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT anomalies_order_chk CHECK (
        first_interval_end <= peak_interval_end AND peak_interval_end <= last_interval_end
    ),
    CONSTRAINT anomalies_p_chk CHECK (min_p IS NULL OR (min_p > 0 AND min_p <= 1)),
    CONSTRAINT anomalies_detectors_chk CHECK (
        cardinality(detectors) >= 1
        AND detectors <@ ARRAY[
            'demand_level', 'demand_residual_short', 'demand_residual_dayahead', 'price_level',
            'price_volatility', 'isolation_forest', 'price_line_300', 'price_line_5000'
        ]::text[]
    ),
    CONSTRAINT anomalies_categories_chk CHECK (
        cardinality(categories) >= 1
        AND categories <@ ARRAY[
            'unusual demand', 'unexpected demand (short-term)', 'unexpected demand (day-ahead)',
            'high price', 'unusual price', 'volatility change', 'multivariate'
        ]::text[]
    ),
    UNIQUE (run_id, first_interval_end)
);
CREATE INDEX anomalies_lookup_idx ON anomalies (region_id, first_interval_end);

ALTER TABLE model_runs DROP CONSTRAINT model_runs_model_kind_check;
ALTER TABLE model_runs ADD CONSTRAINT model_runs_model_kind_check
    CHECK (model_kind IN ('baseline', 'ml', 'interval', 'anomaly'));
ALTER TABLE model_runs DROP CONSTRAINT model_runs_target_check;
ALTER TABLE model_runs ADD CONSTRAINT model_runs_target_check
    CHECK (target IN ('demand', 'price', 'market'));
