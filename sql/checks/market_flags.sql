-- Periods under administered pricing (APC) or market suspension, as contiguous episodes.
-- Informational: these are real market states that must be segmented in evaluation.
WITH flagged AS (
    SELECT interval_end,
           apc_flag,
           market_suspended_flag,
           interval_end - (row_number() OVER (ORDER BY interval_end)) * interval '5 minutes' AS grp
    FROM market_intervals
    WHERE region_id = :region_id
      AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
      AND (apc_flag OR market_suspended_flag)
)
SELECT min(interval_end)           AS first_interval_end,
       max(interval_end)           AS last_interval_end,
       count(*)                    AS n_intervals,
       bool_or(apc_flag)           AS any_apc,
       bool_or(market_suspended_flag) AS any_suspended
FROM flagged
GROUP BY grp
ORDER BY first_interval_end;
