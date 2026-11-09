-- Weather values outside the plausible range for the region's sites
-- (weather.variables.*.plausible in the config). Physically impossible values are already
-- rejected on load; these are flagged for investigation, never clipped.
WITH bounds AS (
    SELECT *
    FROM unnest(
             CAST(:plausible_variable AS text[]),
             CAST(:plausible_lo AS float8[]),
             CAST(:plausible_hi AS float8[])
         ) AS b (variable, lo, hi)
),
grid AS (
    SELECT 'forecast' AS dataset, f.offset_days::int AS offset_days, f.site_id, f.valid_time,
           f.temperature_2m, f.apparent_temperature, f.relative_humidity_2m,
           f.cloud_cover, f.shortwave_radiation, f.wind_speed_10m
    FROM weather_forecasts f
    JOIN weather_sites s USING (site_id)
    WHERE s.region_id = :region_id
      AND f.model = :fc_model
      AND f.valid_time BETWEEN CAST(:weather_first_hour AS timestamptz)
                           AND CAST(:weather_last_hour AS timestamptz)
    UNION ALL
    SELECT 'observation', 0, o.site_id, o.valid_time,
           o.temperature_2m, o.apparent_temperature, o.relative_humidity_2m,
           o.cloud_cover, o.shortwave_radiation, o.wind_speed_10m
    FROM weather_observations o
    JOIN weather_sites s USING (site_id)
    WHERE s.region_id = :region_id
      AND o.model = :obs_model
      AND o.valid_time BETWEEN CAST(:weather_first_hour AS timestamptz)
                           AND CAST(:weather_last_hour AS timestamptz)
)
SELECT g.dataset, g.site_id, g.offset_days, v.variable, g.valid_time, v.value, b.lo, b.hi
FROM grid g
CROSS JOIN LATERAL (
    VALUES ('temperature_2m', g.temperature_2m),
                 ('apparent_temperature', g.apparent_temperature),
                 ('relative_humidity_2m', g.relative_humidity_2m),
                 ('cloud_cover', g.cloud_cover),
                 ('shortwave_radiation', g.shortwave_radiation),
                 ('wind_speed_10m', g.wind_speed_10m)
) AS v (variable, value)
JOIN bounds b USING (variable)
WHERE v.value < b.lo OR v.value > b.hi
ORDER BY g.valid_time, g.dataset, g.site_id, v.variable;
