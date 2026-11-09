-- Months whose most recent ingestion attempt did not succeed (failed or never finished).
WITH latest AS (
    SELECT DISTINCT ON (period_month) period_month, status, error, started_at
    FROM ingestion_runs
    WHERE source = 'mmsdm'
      AND dataset = 'market_intervals'
      AND period_month BETWEEN CAST(:start_month AS date) AND CAST(:end_month AS date)
    ORDER BY period_month, started_at DESC
)
SELECT period_month, status, error, started_at
FROM latest
WHERE status <> 'success'
ORDER BY period_month;
