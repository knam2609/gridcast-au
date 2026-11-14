# M4 reference events: AEMO-flagged intervals

> **Generated** by `gridcast anomaly-reference-events` from the stored `DISPATCHPRICE` flags of VIC1 (`market_intervals.apc_flag_bits`, `market_suspended_flag`). Do not edit by hand. Generated and committed in M4 step 1a, **before any detector code existed**. The rules are in [`docs/plans/m4_plan.md`](plans/m4_plan.md) §3.1 and `configs/base.yaml` (`anomalies.reference_events`).

These flags are AEMO's own record that it modified a dispatch price or suspended the market. They are the only anomaly labels in the data that nobody in this project chose. **They are never detector inputs.** Criterion A2 uses them only to check that the detectors fire when the operator flagged the price.

## What the flags mean

`APCFLAG` is "a 5-bit Region-based field indicating that the original Dispatch Price (ROP) calculated by the Dispatch Algorithm for a region has undergone modification by one of [sic] more of the following processes" (MMS Data Model, `DISPATCHPRICE`):

| Bit value | Meaning (quoted) | Class |
| :--- | :--- | :--- |
| 1 | APC or APF binding (ROP was outside of APC/APF) | P |
| 2 | VoLL Override applied | P |
| 4 | MPC or MPF binding (ROP was outside of MPC/MPF) | P |
| 8 | Price manually overwritten | O |
| 16 | Price Scaling via Inter-regional Loss Factor (IRLF) | O |

- MPC/MPF: market price cap/floor. APC/APF: administered price cap/floor. ROP: the regional price before any capping, scaling or override.
- `MARKETSUSPENDEDFLAG`: "Market suspended flag"; the page gives no further description. Class O.
- Source: the DISPATCHPRICE page of Global-Roam's mirror of AEMO's MMS Data Model, <https://mmsdocs.global-roam.com/MMS/DISPATCH/DISPATCHPRICE/>, read on 2026-11-15. AEMO's own Data Model pages returned HTTP 404 that day. The bit meanings match the comment in `src/gridcast/ingestion/dispatch.py` that the plan was written from, so the classes needed no change.

## Rules

- **Class P** (price-observable by definition): an interval with APCFLAG bit 1, 2 or 4 set, i.e. the price was at a cap, floor or administered limit. Class P wins when an interval also has other flags.
- **Class O**: any other flagged interval, i.e. APCFLAG bit 8 or 16 only, or the market suspended. These may leave no visible trace in Victorian price or demand, so their recall is reported with no claim that anything observable happened.
- **Events:** flagged intervals of one class whose gap is under 24 h form one event. It starts at the start of its first flagged interval and ends at the end of its last.
- **Periods**, by the NEM-time day of the event's start: warm-up before 2022-05-01 (never scored); development 2022-05-01 → 2025-09-30; holdout 2025-10-01 → 2026-09-30.
- **A2** judges every class-P event in development: it passes if each has at least one notable alert from a calibrated detector (the fixed price lines excluded) between 30 min before its start and 30 min after its end.

## Summary

| Class | warm-up | development | holdout | after the holdout | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| P | 2 | 7 | 0 | 0 | 9 |
| O | 1 | 6 | 1 | 0 | 8 |

**A2 judges 7 class-P events:** P03, P04, P05, P06, P07, P08, P09. The holdout has no class-P event, so A2 is judged on development only.

## Class P events

| Event | Period | Start (NEM time) | End (NEM time) | Local start | Duration | Flagged intervals | APCFLAG bits | Suspended intervals | Price range on flagged intervals | Judged by A2 |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | :--- |
| P01 | warm-up | 2022-01-31 17:55 | 2022-01-31 18:00 | 2022-01-31 18:55 AEDT | 5 min | 1 | 4 | 0 | $15,100.00 to $15,100.00 | no |
| P02 | warm-up | 2022-02-21 14:05 | 2022-02-21 14:10 | 2022-02-21 15:05 AEDT | 5 min | 1 | 4 | 0 | −$1,000.00 to −$1,000.00 | no |
| P03 | development | 2022-06-13 17:30 | 2022-06-15 14:00 | 2022-06-13 17:30 AEST | 1 d 20 h 30 min | 335 | 1, 4, 16 | 0 | −$300.00 to $14,570.90 | **yes** |
| P04 | development | 2024-02-13 13:15 | 2024-02-13 15:15 | 2024-02-13 14:15 AEDT | 2 h | 24 | 2, 4 | 0 | $16,600.00 to $16,600.00 | **yes** |
| P05 | development | 2024-08-05 08:00 | 2024-08-05 08:15 | 2024-08-05 08:00 AEST | 15 min | 2 | 4 | 0 | $17,500.00 to $17,500.00 | **yes** |
| P06 | development | 2024-10-28 15:00 | 2024-10-28 15:05 | 2024-10-28 16:00 AEDT | 5 min | 1 | 4 | 0 | −$1,000.00 to −$1,000.00 | **yes** |
| P07 | development | 2024-11-15 11:35 | 2024-11-15 11:40 | 2024-11-15 12:35 AEDT | 5 min | 1 | 4 | 0 | −$1,000.00 to −$1,000.00 | **yes** |
| P08 | development | 2024-11-17 15:45 | 2024-11-17 15:55 | 2024-11-17 16:45 AEDT | 10 min | 2 | 4 | 0 | −$1,000.00 to −$1,000.00 | **yes** |
| P09 | development | 2025-06-12 19:50 | 2025-06-12 19:55 | 2025-06-12 19:50 AEST | 5 min | 1 | 4 | 0 | $17,500.00 to $17,500.00 | **yes** |

## Class O events

| Event | Period | Start (NEM time) | End (NEM time) | Local start | Duration | Flagged intervals | APCFLAG bits | Suspended intervals | Price range on flagged intervals |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| O01 | warm-up | 2022-03-09 03:15 | 2022-03-09 03:20 | 2022-03-09 04:15 AEDT | 5 min | 1 | 8 | 0 | −$29.89 to −$29.89 |
| O02 | development | 2022-06-12 18:50 | 2022-06-24 14:00 | 2022-06-12 18:50 AEST | 11 d 19 h 10 min | 2,766 | 8, 16 | 2,591 | −$55.65 to $533.99 |
| O03 | development | 2022-08-10 11:30 | 2022-08-10 12:35 | 2022-08-10 11:30 AEST | 1 h 5 min | 13 | 8 | 0 | $0.00 to $464.78 |
| O04 | development | 2022-11-17 09:05 | 2022-11-17 09:10 | 2022-11-17 10:05 AEDT | 5 min | 1 | 8 | 0 | $123.09 to $123.09 |
| O05 | development | 2023-03-17 20:10 | 2023-03-17 21:10 | 2023-03-17 21:10 AEDT | 1 h | 12 | 8, 16 | 0 | $108.30 to $113.95 |
| O06 | development | 2023-04-22 16:15 | 2023-04-23 17:00 | 2023-04-22 16:15 AEST | 1 d 45 min | 297 | 8 | 296 | −$44.87 to $168.62 |
| O07 | development | 2024-09-05 13:50 | 2024-09-05 15:10 | 2024-09-05 13:50 AEST | 1 h 20 min | 16 | – | 16 | $34.49 to $53.41 |
| O08 | holdout | 2026-03-23 13:35 | 2026-03-23 18:30 | 2026-03-23 14:35 AEDT | 4 h 55 min | 19 | 16 | 0 | $44.68 to $95.27 |
