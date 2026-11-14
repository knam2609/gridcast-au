# M4 handoff: start here

Read this, then [`m4_plan.md`](m4_plan.md). For the M3 history, see [`m3_handoff.md`](m3_handoff.md) and [`m3_plan.md`](m3_plan.md).

> **Update, 2026-11-15 (fourteenth session: M4 planned):**
> - M3 was merged into `main` on 2026-11-14 (PR #3 → `c5fd5d3`). Branch `m4-anomalies-explainability` is from `main` at `c5fd5d3`.
> - **Three question rounds; the user chose every recommendation** ([`m4_plan.md`](m4_plan.md) §1):
>   - scope: anomaly detection plus SHAP for LightGBM C (price residuals wait for M2b);
>   - causal scoring;
>   - a pre-registered three-part check: alert budget, AEMO-flagged reference events, synthetic injection;
>   - evidence from the cached `DISPATCHREGIONSUM` region totals (no new downloads);
>   - calibrated p-value tiers (notable 0.001, severe 0.0001, trailing 26 weeks) plus fixed price lines at $300 and $5,000;
>   - LightGBM's built-in TreeSHAP over a seeded sample;
>   - descriptive price associations for question 5;
>   - freeze, then score the holdout once;
>   - bars: factor 2 on alert rates, every class-P event caught, ≥ 90% power at each detector's largest injection;
>   - residual detectors on M3's raw QR band with their own p-value.
> - **Added by the plan, accepted by the user on review:** every detector setting is fixed in §2 now, with no tuning spike. The recipe is frozen before the development run, so the development verdicts are also blind.
> - **Disclosed in §3.3:** holdout facts seen during the question rounds (the price calm, 8 Jul 2026 above $5,000, the 6 and 10 Nov 2025 demand jumps, the flagged 23 Mar 2026).
>
> **Next: step 1a.**
> 1. Start Docker (`make db-up`) and check that the SSD is mounted.
> 2. Check the `APCFLAG` and `MARKETSUSPENDEDFLAG` bit meanings in the MMS Data Model (visualisations.aemo.com.au).
> 3. Write `gridcast anomaly-reference-events` and commit the generated `docs/m4_reference_events.md` **before any detector code exists**.
> 4. Then 1b: migration 0007 and `gridcast ingest-generation`.
