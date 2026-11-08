-- AEMO forecast runs whose publication time (LASTCHANGED) is not where the archive layout
-- puts it: pre-dispatch shortly AFTER its run time, 5-minute pre-dispatch shortly BEFORE it
-- (docs/m2_data_spike.md §4-5), or runs with more than one publication time.
WITH runs AS (
    SELECT source, run_time,
           min(available_at) AS first_published,
           max(available_at) AS last_published,
           count(DISTINCT available_at) AS n_published
    FROM aemo_demand_forecasts
    WHERE region_id = :region_id
      AND run_time BETWEEN CAST(:aemo_p5_first AS timestamptz) AND CAST(:aemo_pd_last AS timestamptz)
    GROUP BY source, run_time
)
SELECT source, run_time, first_published, last_published, n_published
FROM runs
WHERE n_published > 1
   OR (source = 'predispatch' AND NOT (
           first_published > run_time
           AND last_published <= run_time + make_interval(mins => CAST(:aemo_pd_within AS int))))
   OR (source = 'p5min' AND NOT (
           last_published < run_time
           AND first_published >= run_time - make_interval(mins => CAST(:aemo_p5_within AS int))))
ORDER BY source, run_time;
