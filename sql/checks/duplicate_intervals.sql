-- Duplicate interval starts for a region. The primary key and unique index make this
-- impossible by construction; the check guards against future schema changes.
SELECT interval_start, count(*) AS n
FROM market_intervals
WHERE region_id = :region_id
GROUP BY interval_start
HAVING count(*) > 1
ORDER BY interval_start;
