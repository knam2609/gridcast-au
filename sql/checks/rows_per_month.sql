-- Row count per NEM-time month versus the expected 288 x days-in-month.
-- 'Australia/Brisbane' is AEST with no daylight saving, i.e. exactly NEM time (UTC+10).
WITH months AS (
    SELECT m::date AS nem_month,
           288 * extract(day FROM (m + interval '1 month' - interval '1 day'))::int AS expected_rows
    FROM generate_series(
             CAST(:start_month AS date),
             CAST(:end_month AS date),
             interval '1 month'
         ) AS m
),
counts AS (
    SELECT date_trunc('month', interval_start AT TIME ZONE 'Australia/Brisbane')::date AS nem_month,
           count(*) AS n_rows
    FROM market_intervals
    WHERE region_id = :region_id
    GROUP BY 1
)
SELECT months.nem_month,
       months.expected_rows,
       coalesce(counts.n_rows, 0)                          AS n_rows,
       months.expected_rows - coalesce(counts.n_rows, 0)   AS shortfall
FROM months
LEFT JOIN counts USING (nem_month)
WHERE coalesce(counts.n_rows, 0) <> months.expected_rows
ORDER BY months.nem_month;
