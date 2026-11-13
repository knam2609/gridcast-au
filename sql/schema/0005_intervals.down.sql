-- Interval runs (and, by cascade, their metrics) cannot satisfy the old constraint.
DELETE FROM model_runs WHERE model_kind = 'interval';
ALTER TABLE model_runs DROP CONSTRAINT model_runs_model_kind_check;
ALTER TABLE model_runs ADD CONSTRAINT model_runs_model_kind_check
    CHECK (model_kind IN ('baseline', 'ml'));
