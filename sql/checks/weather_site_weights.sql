-- The region has no weather sites, or its site weights do not sum to 1 (e.g. a site was
-- removed from the config but is still in weather_sites).
SELECT CAST(:region_id AS text)        AS region_id,
       count(s.site_id)                AS n_sites,
       coalesce(sum(s.weight), 0)      AS weight_sum
FROM weather_sites s
WHERE s.region_id = :region_id
HAVING count(s.site_id) = 0
    OR abs(coalesce(sum(s.weight), 0) - 1) > 1e-9;
