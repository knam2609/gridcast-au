-- Prices below the market floor or above the market price cap of their financial year
-- (FY = 1 July - 30 June, NEM time). Missing cap configuration is also reported.
WITH caps AS (
    SELECT *
    FROM unnest(CAST(:fy_start_years AS int[]), CAST(:fy_caps AS float8[])) AS c (fy_start_year, cap)
),
priced AS (
    SELECT interval_end,
           rrp_aud_mwh,
           extract(year FROM interval_start AT TIME ZONE 'Australia/Brisbane')::int
             - CASE WHEN extract(month FROM interval_start AT TIME ZONE 'Australia/Brisbane') < 7
                    THEN 1 ELSE 0 END AS fy_start_year
    FROM market_intervals
    WHERE region_id = :region_id
      AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
      AND rrp_aud_mwh IS NOT NULL
)
SELECT p.interval_end,
       p.rrp_aud_mwh,
       p.fy_start_year,
       c.cap,
       CASE
           WHEN c.cap IS NULL THEN 'no_cap_configured'
           WHEN p.rrp_aud_mwh > c.cap THEN 'above_cap'
           ELSE 'below_floor'
       END AS reason
FROM priced p
LEFT JOIN caps c USING (fy_start_year)
WHERE c.cap IS NULL
   OR p.rrp_aud_mwh > c.cap
   OR p.rrp_aud_mwh < :price_floor
ORDER BY p.interval_end;
