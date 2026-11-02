-- Forecast rows whose run time or availability time disagrees with the configured rule:
-- run_init = floor(valid_time - offset_days x 24 h) to the run cadence (from 00:00 UTC), and
-- available_at = run_init + publication delay. Guards the leakage-critical timestamps.
WITH expected AS (
    SELECT f.site_id, f.offset_days, f.valid_time, f.run_init, f.available_at,
           date_bin(
               make_interval(hours => CAST(:run_every_hours AS int)),
               f.valid_time - make_interval(hours => 24 * f.offset_days),
               TIMESTAMPTZ '2000-01-01 00:00+00'
           ) AS expected_run_init
    FROM weather_forecasts f
    JOIN weather_sites s USING (site_id)
    WHERE s.region_id = :region_id
      AND f.model = :fc_model
      AND f.valid_time BETWEEN CAST(:weather_first_hour AS timestamptz)
                           AND CAST(:weather_last_hour AS timestamptz)
)
SELECT site_id, offset_days, valid_time, run_init, available_at, expected_run_init,
       expected_run_init + make_interval(secs => CAST(:publication_delay_seconds AS float8))
           AS expected_available_at
FROM expected
WHERE run_init <> expected_run_init
   OR available_at <> expected_run_init
                      + make_interval(secs => CAST(:publication_delay_seconds AS float8))
ORDER BY valid_time, site_id, offset_days;
