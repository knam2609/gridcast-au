-- Prices whose status is not FIRM (e.g. still subject to review).
SELECT coalesce(price_status, '<null>') AS price_status, count(*) AS n
FROM market_intervals
WHERE region_id = :region_id
  AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
  AND price_status IS DISTINCT FROM 'FIRM'
GROUP BY 1
ORDER BY 1;
