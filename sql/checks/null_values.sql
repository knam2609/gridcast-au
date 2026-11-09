-- Intervals present in only one source table (demand without price or vice versa).
SELECT interval_end,
       total_demand_mw IS NULL AS demand_missing,
       rrp_aud_mwh IS NULL     AS price_missing
FROM market_intervals
WHERE region_id = :region_id
  AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
  AND (total_demand_mw IS NULL OR rrp_aud_mwh IS NULL)
ORDER BY interval_end;
