-- How often each documented APCFLAG bit is set (the dispatch price was post-processed).
-- Informational context for price modelling; one row per bit that occurs.
WITH bits (bit_value, meaning) AS (
    VALUES (1, 'administered price cap/floor binding'),
           (2, 'VoLL override'),
           (4, 'market price cap/floor binding'),
           (8, 'price manually overwritten'),
           (16, 'scaled by inter-regional loss factor')
)
SELECT b.bit_value,
       b.meaning,
       count(*)              AS n_intervals,
       min(m.interval_end)   AS first_interval_end,
       max(m.interval_end)   AS last_interval_end
FROM market_intervals m
JOIN bits b ON (m.apc_flag_bits & b.bit_value) <> 0
WHERE m.region_id = :region_id
  AND m.interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
GROUP BY b.bit_value, b.meaning
ORDER BY b.bit_value;
