# M5 handoff: start here

Read this, then [`m5_plan.md`](m5_plan.md). For the M4 history, see [`m4_handoff.md`](m4_handoff.md) and [`m4_plan.md`](m4_plan.md).

> **Update, 2026-11-22 (twenty-third session): M5 planned.** Branch `m5-api` from `main` at `e338672`. Nothing is pushed.
> - **Three question rounds.** The user chose every recommendation except the storage layout:
>   - stored outputs plus an exact recompute;
>   - Postgres serving tables;
>   - issue times 2023-08-01 → 2026-09-30;
>   - price baselines, labelled;
>   - CQR rearranged;
>   - all three extras (per-forecast SHAP, recompute endpoint, forecast evolution);
>   - parity, latency and contract criteria;
>   - local Compose, read-only.
> - **Layout:** after I corrected my size estimate (63.4M rows, ~8 GB, not ~2 GB), the user chose one row per lead for every product over the recommended `real[]` arrays per issue time. I had also wrongly said `forecasts` was empty: it holds a 288-row sample for each of runs 1–80, and the two checked match their parquet exactly. Both corrections are in the plan, §1.
> - **Criteria P1–P6** (plan §3) were fixed before any API code or measurement. They are judged on the Compose service.
> - **Next: step 1a** (migration 0009, `src/gridcast/serving/`, `gridcast publish`), then **1b** (check the Docker VM disk, migrate, publish: ~15–30 min, +8–9 GB).
