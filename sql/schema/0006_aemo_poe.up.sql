-- 0006_aemo_poe: AEMO's probabilistic demand forecasts, every run (milestone 3).
-- A benchmark for the M3 intervals only, never a model input (docs/plans/m3_plan.md §2.6).
--
-- Pre-dispatch PASA (PDPASA_REGIONSOLUTION, MMSDM monthly DATA/ folder), LOR runs only:
-- 48 runs a day, each with POE10, POE50 and POE90 demand (probability of exceedance) for every 30-minute
-- period to the end of the next trading day (docs/m3_data_spike.md).
--
-- Conventions
--   * run_time is the nominal run time (UTC), the run's information cut-off:
--     RUN_DATETIME - 30 min. RUN_DATETIME is the end of the run's first period, and this
--     nominal time matches the pre-dispatch run of the same information (spike §2), so
--     run_time lines up with aemo_demand_forecasts.run_time for source 'predispatch'.
--   * target_end is INTERVAL_DATETIME, the end of the 30-minute period.
--   * available_at is LASTCHANGED, when the run was published (4-7 minutes after run_time).
--   * demand10_mw is exceeded with probability 0.1 (the upper bound of the 80 per cent band),
--     demand90_mw with probability 0.9 (its lower bound).

CREATE TABLE aemo_demand_poe (
    region_id        text NOT NULL REFERENCES regions (region_id),
    run_time         timestamptz NOT NULL,
    target_end       timestamptz NOT NULL,
    available_at     timestamptz NOT NULL,
    demand10_mw      double precision NOT NULL,
    demand50_mw      double precision NOT NULL,
    demand90_mw      double precision NOT NULL,
    ingestion_run_id bigint REFERENCES ingestion_runs (run_id),
    loaded_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (region_id, run_time, target_end),
    CONSTRAINT aemo_poe_grid_chk CHECK (
        date_bin(INTERVAL '30 minutes', target_end, TIMESTAMPTZ '2000-01-01 00:00+00') = target_end
        AND date_bin(INTERVAL '30 minutes', run_time, TIMESTAMPTZ '2000-01-01 00:00+00') = run_time
    ),
    CONSTRAINT aemo_poe_order_chk CHECK (target_end > run_time),
    CONSTRAINT aemo_poe_band_chk CHECK (demand10_mw >= demand50_mw AND demand50_mw >= demand90_mw)
);
