-- 0005_intervals: prediction-interval batches (milestone 3, docs/plans/m3_plan.md).
--
-- An interval batch stores one model_runs row per method and point model, of kind
-- 'interval' (conformal intervals around a stored point forecast; later CQR). Its bounds
-- live in parquet under the data root (artifact_path); forecast_metrics holds coverage,
-- width and interval scores per level and lead (metric names carry the level, e.g.
-- 'coverage_80').

ALTER TABLE model_runs DROP CONSTRAINT model_runs_model_kind_check;
ALTER TABLE model_runs ADD CONSTRAINT model_runs_model_kind_check
    CHECK (model_kind IN ('baseline', 'ml', 'interval'));
