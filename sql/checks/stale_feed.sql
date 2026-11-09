-- The most recent interval is older than the source's freshness threshold (or no data).
SELECT max(interval_end)                         AS latest_interval_end,
       now() - max(interval_end)                 AS age,
       CAST(:stale_after_days AS int)            AS threshold_days
FROM market_intervals
WHERE region_id = :region_id
HAVING max(interval_end) IS NULL
    OR now() - max(interval_end) > make_interval(days => CAST(:stale_after_days AS int));
