-- M4 v2 (docs/plans/m4_plan.md §2.10): v2's D6 is stored as 'robust_distance'. The detector
-- list of 0007 named v1's detectors only; it gains that one name and nothing else changes.
ALTER TABLE anomalies DROP CONSTRAINT anomalies_detectors_chk;
ALTER TABLE anomalies ADD CONSTRAINT anomalies_detectors_chk CHECK (
    cardinality(detectors) >= 1
    AND detectors <@ ARRAY[
        'demand_level', 'demand_residual_short', 'demand_residual_dayahead', 'price_level',
        'price_volatility', 'isolation_forest', 'robust_distance', 'price_line_300',
        'price_line_5000'
    ]::text[]
);
