-- Negative supply values (generation_intervals, M4): output, UIGF, availability and battery
-- load are all non-negative by definition. Grouped by kind and NEM month.
SELECT kind,
       date_trunc('month', interval_end AT TIME ZONE 'Etc/GMT-10' - interval '5 minutes')::date
           AS nem_month,
       count(*) AS n,
       least(min(cleared_mw), min(uigf_mw), min(available_mw), min(load_mw)) AS min_value
FROM generation_intervals
WHERE region_id = :region_id
  AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
  AND (cleared_mw < 0 OR uigf_mw < 0 OR available_mw < 0 OR load_mw < 0)
GROUP BY kind, nem_month
ORDER BY kind, nem_month;
