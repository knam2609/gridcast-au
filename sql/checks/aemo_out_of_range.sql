-- AEMO forecast demand outside the region's plausible bounds (configs/base.yaml).
SELECT source, run_time, target_end, total_demand_mw
FROM aemo_demand_forecasts
WHERE region_id = :region_id
  AND run_time BETWEEN CAST(:aemo_p5_first AS timestamptz) AND CAST(:aemo_pd_last AS timestamptz)
  AND (total_demand_mw < :demand_min OR total_demand_mw > :demand_max)
ORDER BY source, run_time, target_end;
