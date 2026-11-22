-- Semi-scheduled wind or solar cleared above its UIGF or its availability by more than the
-- tolerance (generation_intervals, M4). Informational: the values are AEMO's as published.
-- Grouped by kind and NEM month.
SELECT kind,
       date_trunc('month', interval_end AT TIME ZONE 'Etc/GMT-10' - interval '5 minutes')::date
           AS nem_month,
       count(*) AS n_intervals,
       count(*) FILTER (WHERE cleared_mw > uigf_mw + :tolerance_mw) AS above_uigf,
       count(*) FILTER (WHERE cleared_mw > available_mw + :tolerance_mw) AS above_available,
       max(cleared_mw - least(uigf_mw, coalesce(available_mw, uigf_mw))) AS max_excess_mw
FROM generation_intervals
WHERE region_id = :region_id
  AND kind IN ('wind', 'solar')
  AND interval_end BETWEEN CAST(:window_start AS timestamptz) AND CAST(:window_end AS timestamptz)
  AND (cleared_mw > uigf_mw + :tolerance_mw OR cleared_mw > available_mw + :tolerance_mw)
GROUP BY kind, nem_month
ORDER BY kind, nem_month;
