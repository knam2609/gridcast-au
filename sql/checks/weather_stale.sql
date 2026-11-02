-- Weather series (per site and offset) that end before the hour needed by the region's latest
-- market interval, so features for the newest intervals would have no weather.
WITH latest_market AS (
    SELECT max(interval_end) AS interval_end
    FROM market_intervals
    WHERE region_id = :region_id
),
needed AS (
    -- ceil to the hour: an interval ending at t needs the hourly value at ceil_1h(t)
    SELECT date_bin('1 hour', interval_end - interval '1 microsecond',
                    TIMESTAMPTZ '2000-01-01 00:00+00') + interval '1 hour' AS needed_hour
    FROM latest_market
    WHERE interval_end IS NOT NULL
),
sites AS (
    SELECT site_id FROM weather_sites WHERE region_id = :region_id
),
latest AS (
    SELECT 'forecast' AS dataset, s.site_id, k.offset_days,
           (SELECT max(f.valid_time) FROM weather_forecasts f
             WHERE f.model = :fc_model AND f.site_id = s.site_id
               AND f.offset_days = k.offset_days) AS latest_valid_time
    FROM sites s
    CROSS JOIN unnest(CAST(:fc_offsets AS int[])) AS k (offset_days)
    UNION ALL
    SELECT 'observation', s.site_id, 0,
           (SELECT max(o.valid_time) FROM weather_observations o
             WHERE o.model = :obs_model AND o.site_id = s.site_id)
    FROM sites s
)
SELECT l.dataset, l.site_id, l.offset_days, l.latest_valid_time, n.needed_hour
FROM latest l
CROSS JOIN needed n
WHERE l.latest_valid_time IS NULL OR l.latest_valid_time < n.needed_hour
ORDER BY l.dataset, l.site_id, l.offset_days;
