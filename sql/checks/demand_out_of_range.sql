-- Demand outside the region's plausible range (negative demand is impossible for VIC
-- TOTALDEMAND; the upper bound sits well above the all-time record).
SELECT interval_end, total_demand_mw
FROM market_intervals
WHERE region_id = :region_id
  AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
  AND (total_demand_mw < :demand_min OR total_demand_mw > :demand_max)
ORDER BY interval_end;
