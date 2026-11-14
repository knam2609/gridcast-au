-- Missing supply rows (generation_intervals, M4), collapsed into contiguous gaps per kind.
-- Wind, solar and dispatchable generation are expected at every 5-minute interval of the
-- window; battery (bidirectional units) from its first loaded interval in the window on.
WITH kinds AS (
    SELECT k.kind,
           CASE
               WHEN k.kind = 'battery' THEN (
                   SELECT min(interval_end)
                   FROM generation_intervals
                   WHERE region_id = :region_id
                     AND kind = 'battery'
                     AND interval_end BETWEEN CAST(:window_start AS timestamptz)
                                          AND CAST(:window_end AS timestamptz)
               )
               ELSE CAST(:window_start AS timestamptz)
           END AS first_end
    FROM (VALUES ('wind'), ('solar'), ('dispatchable'), ('battery')) AS k (kind)
),
expected AS (
    SELECT k.kind,
           generate_series(k.first_end, CAST(:window_end AS timestamptz), interval '5 minutes')
               AS interval_end
    FROM kinds k
    WHERE k.first_end IS NOT NULL
),
missing AS (
    SELECT e.kind, e.interval_end
    FROM expected e
    LEFT JOIN generation_intervals g
           ON g.region_id = :region_id
          AND g.kind = e.kind
          AND g.interval_end = e.interval_end
    WHERE g.interval_end IS NULL
),
islands AS (
    SELECT kind,
           interval_end,
           interval_end
               - (row_number() OVER (PARTITION BY kind ORDER BY interval_end)) * interval '5 minutes'
               AS grp
    FROM missing
)
SELECT kind,
       min(interval_end) AS gap_first_interval_end,
       max(interval_end) AS gap_last_interval_end,
       count(*)          AS n_missing
FROM islands
GROUP BY kind, grp
ORDER BY kind, gap_first_interval_end;
