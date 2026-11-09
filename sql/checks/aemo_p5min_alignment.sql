-- Timestamp-alignment guard: interval 0 of a 5-minute pre-dispatch run ends at the run time
-- and should equal that interval's dispatch demand (market_intervals.total_demand_mw).
-- Measured over Sep 2024 -> Sep 2026: aligned, at most 7.25% of a month's values differ
-- (mean absolute difference <= 0.32 MW, cause unverified); shifted by one interval either
-- way, >= 99.97% differ (38-124 MW). Flags NEM-time months where over half differ.
WITH i0 AS (
    SELECT date_trunc('month', f.run_time AT TIME ZONE 'Etc/GMT-10')::date AS nem_month,
           abs(f.total_demand_mw - m.total_demand_mw) > 0.01 AS differs,
           abs(f.total_demand_mw - m.total_demand_mw) AS abs_diff
    FROM aemo_demand_forecasts f
    JOIN market_intervals m
      ON m.region_id = f.region_id AND m.interval_end = f.target_end
    WHERE f.region_id = :region_id
      AND f.source = 'p5min'
      AND f.target_end = f.run_time
      AND f.run_time BETWEEN CAST(:aemo_p5_first AS timestamptz) AND CAST(:aemo_p5_last AS timestamptz)
)
SELECT nem_month, count(*) AS n_runs,
       count(*) FILTER (WHERE differs) AS n_differ,
       round(100.0 * count(*) FILTER (WHERE differs) / count(*), 2) AS pct_differ,
       avg(abs_diff) AS mean_abs_diff_mw
FROM i0
GROUP BY nem_month
HAVING count(*) FILTER (WHERE differs) > 0.5 * count(*)
ORDER BY nem_month;
