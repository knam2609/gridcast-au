-- Reverts 0001_core.
DROP VIEW IF EXISTS v_market_intervals_local;
DROP TABLE IF EXISTS dq_results;
DROP TABLE IF EXISTS forecasts;
DROP TABLE IF EXISTS forecast_metrics;
DROP TABLE IF EXISTS model_runs;
DROP TABLE IF EXISTS calendar_days;
DROP TABLE IF EXISTS market_intervals;
DROP TABLE IF EXISTS ingestion_runs;
DROP TABLE IF EXISTS regions;
