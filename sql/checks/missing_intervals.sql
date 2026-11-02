-- Missing 5-minute intervals within the configured window, collapsed into contiguous gaps
-- (gaps-and-islands: consecutive missing timestamps share the same `grp`).
WITH expected AS (
    SELECT generate_series(
               CAST(:window_start AS timestamptz),
               CAST(:window_end AS timestamptz),
               interval '5 minutes'
           ) AS interval_end
),
missing AS (
    SELECT e.interval_end
    FROM expected e
    LEFT JOIN market_intervals m
           ON m.region_id = :region_id
          AND m.interval_end = e.interval_end
    WHERE m.interval_end IS NULL
),
islands AS (
    SELECT interval_end,
           interval_end - (row_number() OVER (ORDER BY interval_end)) * interval '5 minutes' AS grp
    FROM missing
)
SELECT min(interval_end) AS gap_first_interval_end,
       max(interval_end) AS gap_last_interval_end,
       count(*)          AS n_missing
FROM islands
GROUP BY grp
ORDER BY gap_first_interval_end;
