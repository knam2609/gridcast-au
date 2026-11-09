-- Reverts 0002_price_flag_codes (lossy: non-zero codes collapse to true).
DROP VIEW v_market_intervals_local;

ALTER TABLE market_intervals
    DROP COLUMN administered_price_binding,
    DROP COLUMN market_cap_floor_binding,
    DROP COLUMN price_manually_overwritten,
    DROP COLUMN market_suspended,
    DROP CONSTRAINT market_intervals_apc_bits_chk,
    DROP CONSTRAINT market_intervals_suspended_chk;

ALTER TABLE market_intervals RENAME COLUMN apc_flag_bits TO apc_flag;
ALTER TABLE market_intervals
    ALTER COLUMN apc_flag DROP DEFAULT,
    ALTER COLUMN apc_flag TYPE boolean USING (apc_flag <> 0),
    ALTER COLUMN apc_flag SET DEFAULT false,
    ALTER COLUMN market_suspended_flag DROP DEFAULT,
    ALTER COLUMN market_suspended_flag TYPE boolean USING (market_suspended_flag <> 0),
    ALTER COLUMN market_suspended_flag SET DEFAULT false;

CREATE VIEW v_market_intervals_local AS
SELECT
    m.region_id,
    m.interval_start,
    m.interval_end,
    (m.interval_start AT TIME ZONE r.local_tz)                         AS local_start,
    (m.interval_start AT TIME ZONE r.local_tz)::date                   AS local_date,
    extract(hour FROM m.interval_start AT TIME ZONE r.local_tz)::int   AS local_hour,
    extract(isodow FROM m.interval_start AT TIME ZONE r.local_tz)::int AS local_isodow,
    c.is_weekend,
    c.is_public_holiday,
    c.holiday_name,
    c.is_dst_change_day,
    m.total_demand_mw,
    m.rrp_aud_mwh,
    m.net_interchange_mw,
    m.available_generation_mw,
    m.apc_flag,
    m.market_suspended_flag
FROM market_intervals m
JOIN regions r USING (region_id)
LEFT JOIN calendar_days c
       ON c.jurisdiction = r.holiday_jurisdiction
      AND c.local_date = (m.interval_start AT TIME ZONE r.local_tz)::date;
