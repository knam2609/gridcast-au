-- Hourly weather values missing inside the window, per region site, collapsed into contiguous
-- gaps per (dataset, site, offset, variable). A missing row counts as missing values.
-- Each gap is classified against the known/expected gaps from the config
-- (weather.forecasts.expected_missing, step 0). :want_known selects which kind is returned:
-- weather_gaps (fail) returns unexpected gaps, weather_known_gaps (warn) the expected ones.
-- Observations (ERA5) have no expected gaps; their offset_days is reported as 0.
WITH hours AS (
    SELECT generate_series(
               CAST(:weather_first_hour AS timestamptz),
               CAST(:weather_last_hour AS timestamptz),
               interval '1 hour'
           ) AS valid_time
),
sites AS (
    SELECT site_id FROM weather_sites WHERE region_id = :region_id
),
grid AS (
    SELECT 'forecast' AS dataset, k.offset_days, s.site_id, h.valid_time,
           f.temperature_2m, f.apparent_temperature, f.relative_humidity_2m,
           f.cloud_cover, f.shortwave_radiation, f.wind_speed_10m
    FROM unnest(CAST(:fc_offsets AS int[])) AS k (offset_days)
    CROSS JOIN sites s
    CROSS JOIN hours h
    LEFT JOIN weather_forecasts f
           ON f.model = :fc_model
          AND f.site_id = s.site_id
          AND f.offset_days = k.offset_days
          AND f.valid_time = h.valid_time
    UNION ALL
    SELECT 'observation', 0, s.site_id, h.valid_time,
           o.temperature_2m, o.apparent_temperature, o.relative_humidity_2m,
           o.cloud_cover, o.shortwave_radiation, o.wind_speed_10m
    FROM sites s
    CROSS JOIN hours h
    LEFT JOIN weather_observations o
           ON o.model = :obs_model
          AND o.site_id = s.site_id
          AND o.valid_time = h.valid_time
),
missing AS (
    SELECT g.dataset, g.offset_days, g.site_id, g.valid_time, v.variable
    FROM grid g
    CROSS JOIN LATERAL (
        VALUES ('temperature_2m', g.temperature_2m),
                 ('apparent_temperature', g.apparent_temperature),
                 ('relative_humidity_2m', g.relative_humidity_2m),
                 ('cloud_cover', g.cloud_cover),
                 ('shortwave_radiation', g.shortwave_radiation),
                 ('wind_speed_10m', g.wind_speed_10m)
    ) AS v (variable, value)
    WHERE v.value IS NULL
),
known AS (
    SELECT *
    FROM unnest(
             CAST(:known_reason AS text[]),
             CAST(:known_variable AS text[]),
             CAST(:known_offset AS int[]),
             CAST(:known_first AS timestamptz[]),
             CAST(:known_last AS timestamptz[])
         ) AS k (reason, variable, offset_days, first_hour, last_hour)
),
classified AS (
    SELECT m.*, kn.reason AS known_reason
    FROM missing m
    LEFT JOIN LATERAL (
        SELECT k.reason
        FROM known k
        WHERE m.dataset = 'forecast'
          AND k.variable = m.variable
          AND k.offset_days = m.offset_days
          AND m.valid_time BETWEEN k.first_hour AND k.last_hour
        ORDER BY k.reason
        LIMIT 1
    ) kn ON true
),
islands AS (
    SELECT c.*,
           c.valid_time - (row_number() OVER (
               PARTITION BY dataset, site_id, offset_days, variable, known_reason
               ORDER BY valid_time
           )) * interval '1 hour' AS grp
    FROM classified c
    WHERE (c.known_reason IS NOT NULL) = CAST(:want_known AS boolean)
)
SELECT dataset,
       site_id,
       offset_days,
       variable,
       known_reason,
       min(valid_time) AS first_missing,
       max(valid_time) AS last_missing,
       count(*)        AS n_missing
FROM islands
GROUP BY dataset, site_id, offset_days, variable, known_reason, grp
ORDER BY dataset, offset_days, variable, site_id, first_missing;
