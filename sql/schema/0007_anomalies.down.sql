DROP TABLE anomalies;
DROP TABLE generation_intervals;
-- Anomaly runs and market-target runs (and, by cascade, their metrics) cannot satisfy the
-- old constraints.
DELETE FROM model_runs WHERE model_kind = 'anomaly' OR target = 'market';
ALTER TABLE model_runs DROP CONSTRAINT model_runs_target_check;
ALTER TABLE model_runs ADD CONSTRAINT model_runs_target_check
    CHECK (target IN ('demand', 'price'));
ALTER TABLE model_runs DROP CONSTRAINT model_runs_model_kind_check;
ALTER TABLE model_runs ADD CONSTRAINT model_runs_model_kind_check
    CHECK (model_kind IN ('baseline', 'ml', 'interval'));
