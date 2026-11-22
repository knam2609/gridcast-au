-- Timestamps that are off the 5-minute grid, have the wrong duration, lie in the future,
-- or fall outside the configured ingestion window.
-- Days after the window that were loaded from the NEMWeb daily reports (a successful
-- 'nemweb_daily' run, m2_plan.md step 6) are there on purpose, so they are not flagged as
-- outside the window; daily_report_days lists them instead. Grid, duration and future
-- checks still apply to them.
WITH tagged AS (
    SELECT m.interval_start,
           m.interval_end,
           coalesce(r.source = 'nemweb_daily' AND r.status = 'success', false) AS from_daily
    FROM market_intervals m
    LEFT JOIN ingestion_runs r ON r.run_id = m.ingestion_run_id
    WHERE m.region_id = :region_id
)
SELECT interval_start,
       interval_end,
       CASE
           WHEN interval_end - interval_start <> interval '5 minutes' THEN 'bad_duration'
           WHEN mod(extract(epoch FROM interval_end)::bigint, 300) <> 0 THEN 'off_grid'
           WHEN interval_end > now() THEN 'in_future'
           ELSE 'outside_window'
       END AS reason
FROM tagged
WHERE interval_end - interval_start <> interval '5 minutes'
   OR mod(extract(epoch FROM interval_end)::bigint, 300) <> 0
   OR interval_end > now()
   OR interval_end < CAST(:window_start AS timestamptz)
   OR (interval_end > CAST(:window_end AS timestamptz) AND NOT from_daily)
ORDER BY interval_end;
