-- Market days after the configured window that were loaded from the NEMWeb daily reports
-- (a successful 'nemweb_daily' run, m2_plan.md step 6), one row per NEM day (from the
-- interval start). They are there on purpose and implausible_timestamps does not flag them;
-- this warning keeps them visible. Once a day is re-ingested from MMSDM it leaves this list,
-- and implausible_timestamps flags it unless ingestion.end_month is extended.
SELECT (m.interval_start AT TIME ZONE 'Etc/GMT-10')::date AS nem_day,
       m.ingestion_run_id,
       count(*) AS n_intervals,
       min(m.interval_end) AS first_interval_end,
       max(m.interval_end) AS last_interval_end
FROM market_intervals m
JOIN ingestion_runs r ON r.run_id = m.ingestion_run_id
WHERE m.region_id = :region_id
  AND r.source = 'nemweb_daily'
  AND r.status = 'success'
  AND m.interval_end > CAST(:window_end AS timestamptz)
GROUP BY 1, 2
ORDER BY 1, 2;
