-- Duplicate (region, interval) records found in the *source files* and removed during ETL
-- (latest successful ingestion run per month). Informational: AEMO re-issues can do this.
WITH latest AS (
    SELECT DISTINCT ON (period_month) period_month, duplicates_dropped
    FROM ingestion_runs
    WHERE source = 'mmsdm'
      AND dataset = 'market_intervals'
      AND status = 'success'
      AND period_month BETWEEN CAST(:start_month AS date) AND CAST(:end_month AS date)
    ORDER BY period_month, started_at DESC
)
SELECT period_month, duplicates_dropped
FROM latest
WHERE duplicates_dropped > 0
ORDER BY period_month;
