-- Months whose most recent ingestion attempt did not succeed (failed or never finished),
-- for every source and dataset (AEMO market data, Open-Meteo weather, ...).
WITH latest AS (
    SELECT DISTINCT ON (source, dataset, period_month)
           source, dataset, period_month, status, error, started_at
    FROM ingestion_runs
    WHERE period_month BETWEEN CAST(:start_month AS date) AND CAST(:end_month AS date)
    ORDER BY source, dataset, period_month, started_at DESC
)
SELECT source, dataset, period_month, status, error, started_at
FROM latest
WHERE status <> 'success'
ORDER BY source, dataset, period_month;
