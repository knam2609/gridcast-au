-- Weather values clipped onto their physical range on load (rounding artefacts within
-- weather.clip_tolerance, e.g. GFS cloud cover of 101 %), per site, month and variable, from
-- the latest successful load of each month (ingestion_runs.files[].clipped_values).
WITH latest AS (
    SELECT DISTINCT ON (dataset, period_month) dataset, period_month, files
    FROM ingestion_runs
    WHERE source = 'open_meteo'
      AND status = 'success'
      AND period_month BETWEEN CAST(:start_month AS date) AND CAST(:end_month AS date)
    ORDER BY dataset, period_month, started_at DESC
)
SELECT l.dataset,
       l.period_month,
       f ->> 'site_id'   AS site_id,
       c.key             AS variable,
       c.value::int      AS n_clipped
FROM latest l
CROSS JOIN LATERAL jsonb_array_elements(l.files) AS f
CROSS JOIN LATERAL jsonb_each_text(coalesce(f -> 'clipped_values', '{}'::jsonb)) AS c
WHERE c.value::int > 0
ORDER BY l.dataset, l.period_month, site_id, variable;
