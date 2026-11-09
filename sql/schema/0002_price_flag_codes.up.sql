-- 0002_price_flag_codes: DISPATCHPRICE.APCFLAG is a 5-bit field, not a boolean, and
-- MARKETSUSPENDEDFLAG is NUMBER(3,0). Store the raw codes and expose each documented
-- meaning as a generated boolean column (MMS Data Model Report, table DISPATCHPRICE):
--   bit value 1  = Administered Price Cap/Floor binding (ROP outside APC/APF range)
--   bit value 2  = VoLL override applied
--   bit value 4  = Market Price Cap/Floor binding (ROP outside MPC/MPF range)
--   bit value 8  = price manually overwritten
--   bit value 16 = price scaled by the inter-regional loss factor (IRLF)

DROP VIEW v_market_intervals_local;

ALTER TABLE market_intervals
    ALTER COLUMN apc_flag DROP DEFAULT,
    ALTER COLUMN apc_flag TYPE smallint USING (CASE WHEN apc_flag THEN 1 ELSE 0 END),
    ALTER COLUMN apc_flag SET DEFAULT 0;
ALTER TABLE market_intervals RENAME COLUMN apc_flag TO apc_flag_bits;
ALTER TABLE market_intervals
    ADD CONSTRAINT market_intervals_apc_bits_chk CHECK (apc_flag_bits BETWEEN 0 AND 31);

ALTER TABLE market_intervals
    ALTER COLUMN market_suspended_flag DROP DEFAULT,
    ALTER COLUMN market_suspended_flag TYPE smallint
        USING (CASE WHEN market_suspended_flag THEN 1 ELSE 0 END),
    ALTER COLUMN market_suspended_flag SET DEFAULT 0;
ALTER TABLE market_intervals
    ADD CONSTRAINT market_intervals_suspended_chk CHECK (market_suspended_flag >= 0);

ALTER TABLE market_intervals
    ADD COLUMN administered_price_binding boolean
        GENERATED ALWAYS AS ((apc_flag_bits & 1) <> 0) STORED,
    ADD COLUMN market_cap_floor_binding boolean
        GENERATED ALWAYS AS ((apc_flag_bits & 4) <> 0) STORED,
    ADD COLUMN price_manually_overwritten boolean
        GENERATED ALWAYS AS ((apc_flag_bits & 8) <> 0) STORED,
    ADD COLUMN market_suspended boolean
        GENERATED ALWAYS AS (market_suspended_flag <> 0) STORED;

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
    m.apc_flag_bits,
    m.administered_price_binding,
    m.market_cap_floor_binding,
    m.price_manually_overwritten,
    m.market_suspended
FROM market_intervals m
JOIN regions r USING (region_id)
LEFT JOIN calendar_days c
       ON c.jurisdiction = r.holiday_jurisdiction
      AND c.local_date = (m.interval_start AT TIME ZONE r.local_tz)::date;
