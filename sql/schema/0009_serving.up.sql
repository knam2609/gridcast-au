-- 0009_serving: what the M5 API serves (docs/plans/m5_plan.md §2.2).
--
-- `gridcast publish` loads the scored outputs of milestones 1-3 into Postgres, one row per
-- (run, issue time, target), and checks every value against its parquet source:
--   * point forecasts (LightGBM C demand, the M1 price baselines) go into `forecasts`
--     (point only; lower, upper and interval_level stay NULL);
--   * demand intervals (CQR, rearranged, central levels 50, 80 and 95) go into
--     `forecast_intervals`;
--   * every (product, split) publication is recorded in `publications`.

CREATE TABLE forecast_intervals (
    run_id     bigint NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    region_id  text NOT NULL REFERENCES regions (region_id),
    issue_time timestamptz NOT NULL,
    target_end timestamptz NOT NULL,
    horizon    smallint NOT NULL CHECK (horizon >= 1),
    lo95       real NOT NULL,
    lo80       real NOT NULL,
    lo50       real NOT NULL,
    hi50       real NOT NULL,
    hi80       real NOT NULL,
    hi95       real NOT NULL,
    PRIMARY KEY (run_id, issue_time, target_end),
    CONSTRAINT forecast_intervals_future_chk CHECK (target_end > issue_time),
    -- The levels nest: the database itself rejects a non-nested row.
    CONSTRAINT forecast_intervals_nested_chk CHECK (
        lo95 <= lo80 AND lo80 <= lo50 AND lo50 <= hi50 AND hi50 <= hi80 AND hi80 <= hi95
    ),
    -- Every bound is finite. Postgres orders NaN above every number, so with the chain
    -- above a NaN anywhere forces hi95 = NaN, which fails hi95 < Infinity.
    CONSTRAINT forecast_intervals_finite_chk CHECK (
        lo95 > '-Infinity'::real AND hi95 < 'Infinity'::real
    )
);

-- One row per (product, split) publication attempt; /v1/meta and /health read the latest
-- successful one of each.
CREATE TABLE publications (
    publication_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product        text NOT NULL CHECK (product IN (
                       'demand_point', 'demand_interval', 'price_persistence',
                       'price_median_profile'
                   )),
    split          text NOT NULL CHECK (split IN ('dev', 'holdout')),
    region_id      text NOT NULL REFERENCES regions (region_id),
    run_id         bigint NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    source_run_id  bigint NOT NULL REFERENCES model_runs (run_id),  -- = run_id except intervals
    source_files   jsonb NOT NULL,                  -- [{path, sha256, bytes}] under the data root
    first_issue    timestamptz,
    last_issue     timestamptz,
    issue_times    integer,
    rows_expected  bigint,                          -- issue times x leads - missing pairs
    missing_pairs  bigint,                          -- pairs without a stored value (not published)
    rows_inserted  bigint,                          -- by this attempt (0 on an idempotent rerun)
    git_sha        text,
    status         text NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running', 'success', 'failed')),
    checks         jsonb NOT NULL DEFAULT '{}',     -- P1 (and P4 for intervals), timings
    error          text,
    started_at     timestamptz NOT NULL DEFAULT now(),
    published_at   timestamptz,
    CONSTRAINT publications_order_chk CHECK (first_issue IS NULL OR first_issue <= last_issue)
);
CREATE INDEX publications_lookup_idx ON publications (product, split, started_at DESC);
