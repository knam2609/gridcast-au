-- 0004_aemo_forecasts: AEMO's own regional demand forecasts, every run (milestone 2).
-- A benchmark only, never a model input (docs/plans/m2_plan.md §2.6).
--
--   source = 'predispatch'  30-minute pre-dispatch (PREDISPATCHREGIONSUM, 48 runs a day).
--                           total_demand_mw is the mean over the 30 minutes ending target_end.
--   source = 'p5min'        5-minute pre-dispatch (P5MIN_REGIONSOLUTION, 288 runs a day).
--                           Interval 0 ends at run_time and equals the dispatch demand.
--
-- Conventions
--   * run_time is the nominal run time (UTC), i.e. the run's information cut-off:
--     pre-dispatch 04:00 NEM time + (PP - 1) x 30 min on the trading day of
--     PREDISPATCHSEQNO; 5-minute pre-dispatch RUN_DATETIME.
--   * available_at is LASTCHANGED, when the run was published.
--   * Only the pricing run (INTERVENTION = 0) is stored.

CREATE TABLE aemo_demand_forecasts (
    source             text NOT NULL CHECK (source IN ('predispatch', 'p5min')),
    region_id          text NOT NULL REFERENCES regions (region_id),
    run_time           timestamptz NOT NULL,
    target_end         timestamptz NOT NULL,             -- end of the forecast period
    resolution_minutes smallint NOT NULL,
    available_at       timestamptz NOT NULL,             -- LASTCHANGED
    total_demand_mw    double precision NOT NULL,
    ingestion_run_id   bigint REFERENCES ingestion_runs (run_id),
    loaded_at          timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, region_id, run_time, target_end),
    CONSTRAINT aemo_forecasts_resolution_chk CHECK (
        (source = 'predispatch' AND resolution_minutes = 30)
        OR (source = 'p5min' AND resolution_minutes = 5)
    ),
    CONSTRAINT aemo_forecasts_grid_chk CHECK (
        date_bin(make_interval(mins => resolution_minutes), target_end,
                 TIMESTAMPTZ '2000-01-01 00:00+00') = target_end
        AND date_bin(make_interval(mins => resolution_minutes), run_time,
                     TIMESTAMPTZ '2000-01-01 00:00+00') = run_time
    ),
    -- Pre-dispatch periods end after the run time; 5-minute interval 0 ends at it.
    CONSTRAINT aemo_forecasts_order_chk CHECK (
        target_end > run_time OR (source = 'p5min' AND target_end = run_time)
    )
);
