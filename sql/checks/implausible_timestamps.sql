-- Timestamps that are off the 5-minute grid, have the wrong duration, lie in the future,
-- or fall outside the configured ingestion window.
SELECT interval_start,
       interval_end,
       CASE
           WHEN interval_end - interval_start <> interval '5 minutes' THEN 'bad_duration'
           WHEN mod(extract(epoch FROM interval_end)::bigint, 300) <> 0 THEN 'off_grid'
           WHEN interval_end > now() THEN 'in_future'
           ELSE 'outside_window'
       END AS reason
FROM market_intervals
WHERE region_id = :region_id
  AND (   interval_end - interval_start <> interval '5 minutes'
       OR mod(extract(epoch FROM interval_end)::bigint, 300) <> 0
       OR interval_end > now()
       OR interval_end < CAST(:window_start AS timestamptz)
       OR interval_end > CAST(:window_end AS timestamptz))
ORDER BY interval_end;
