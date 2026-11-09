-- Episodes of administered pricing (APCFLAG bit 1) or market suspension, as contiguous
-- runs of intervals. Informational: these are real market states that must be segmented
-- in evaluation, not errors in the data.
WITH flagged AS (
    SELECT interval_end,
           administered_price_binding,
           market_suspended,
           interval_end - (row_number() OVER (ORDER BY interval_end)) * interval '5 minutes' AS grp
    FROM market_intervals
    WHERE region_id = :region_id
      AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
      AND (administered_price_binding OR market_suspended)
)
SELECT min(interval_end)               AS first_interval_end,
       max(interval_end)               AS last_interval_end,
       count(*)                        AS n_intervals,
       bool_or(administered_price_binding) AS any_administered_price,
       bool_or(market_suspended)       AS any_suspended
FROM flagged
GROUP BY grp
ORDER BY first_interval_end;
