-- Events naming v2's D6 cannot satisfy 0007's detector list.
DELETE FROM anomalies WHERE 'robust_distance' = ANY (detectors);
ALTER TABLE anomalies DROP CONSTRAINT anomalies_detectors_chk;
ALTER TABLE anomalies ADD CONSTRAINT anomalies_detectors_chk CHECK (
    cardinality(detectors) >= 1
    AND detectors <@ ARRAY[
        'demand_level', 'demand_residual_short', 'demand_residual_dayahead', 'price_level',
        'price_volatility', 'isolation_forest', 'price_line_300', 'price_line_5000'
    ]::text[]
);
