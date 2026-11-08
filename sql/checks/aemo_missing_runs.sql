-- AEMO forecast runs missing from the expected cadence (pre-dispatch every 30 min from
-- 04:00 NEM time on the first trading day; 5-minute pre-dispatch every 5 min), per source
-- and NEM-time month. A missing run leaves its issue times without an AEMO comparison.
WITH expected AS (
    SELECT 'predispatch' AS source, t AS run_time
    FROM generate_series(CAST(:aemo_pd_first AS timestamptz), CAST(:aemo_pd_last AS timestamptz),
                         interval '30 minutes') AS t
    UNION ALL
    SELECT 'p5min', t
    FROM generate_series(CAST(:aemo_p5_first AS timestamptz), CAST(:aemo_p5_last AS timestamptz),
                         interval '5 minutes') AS t
),
present AS (
    SELECT DISTINCT source, run_time
    FROM aemo_demand_forecasts
    WHERE region_id = :region_id
)
SELECT e.source,
       date_trunc('month', e.run_time AT TIME ZONE 'Etc/GMT-10')::date AS nem_month,
       count(*) AS n_missing,
       min(e.run_time) AS first_missing,
       max(e.run_time) AS last_missing
FROM expected e
LEFT JOIN present p USING (source, run_time)
WHERE p.run_time IS NULL
GROUP BY 1, 2
ORDER BY 1, 2;
