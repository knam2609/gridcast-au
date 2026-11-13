# M3 handoff: start here

Read this, then [`m3_plan.md`](m3_plan.md). For the M2 history, see [`m2_handoff.md`](m2_handoff.md) and [`m2_plan.md`](m2_plan.md).

> **Update, 2026-11-13 (thirteenth session, step 5: half-hour intervals, the freeze, the AEMO comparison):**
> - **Question round (user chose every recommendation; `m3_plan.md` step 5):** build the 30-minute recalibration first, then freeze both resolutions, then load PDPASA; freeze the rearranged row; CQR, conformal-around-C and the best baseline against AEMO; M2's report layout.
> - **`bc5337e` (5a):** `gridcast intervals-halfhour` and `gridcast intervals-freeze-check`. Half-hour batch **`27aa78d6-8ab8-4509-aa40-d8b08cda68ea`** (runs 87–91; CQR 80% coverage 79.4%, reported only). The freeze check rebuilt the step-1 and step-3 batches exactly from `bc5337e`.
> - **5b:** the freeze record in `configs/base.yaml` (`intervals.frozen_*`: hashes `d34273400899af69` and `8767a8948966637c`, commit `bc5337e`), pinned by unit tests.
> - **`526f048` (5c):** migration 0006 (`aemo_demand_poe`), `gridcast ingest-aemo-poe` and `gridcast report-m3-aemo`, written after the freeze and before the load. PDPASA Sep 2024 → Sep 2026 loaded (1.1 GB cached, under 2 minutes).
> - **5d:** [`reports/m3_aemo.md`](../../reports/m3_aemo.md), generated from `526f048`. Reported only:
>   - AEMO's band (matched) covers 70.9% of actual demand, CQR 80.2%.
>   - CQR's interval is about twice as wide, and its IS80 is higher (+395.8 MW, CI +348.8 to +444.4). CQR wins only at leads up to 2 h, consistent with M2's point comparison.
>
> **Next:** step 6, the holdout scored once (`gridcast evaluate-holdout-intervals`, U5 lock; `m3_plan.md` step 6). Run a question round first. Open points:
> - how the holdout run builds conformal and CQR intervals whose calibration windows reach back into development data (combine run 2's and `holdout_main`'s stored forecasts);
> - a reproduction check for the interval code, beyond QR's Aug–Sep 2025 forecasts (e.g. the development bounds of Aug–Sep 2025 recomputed exactly);
> - the 5-minute and half-hour holdout batches, plus the AEMO holdout part;
> - QR holdout compute: projected 4–6 h from a worktree.
>
> **Update, 2026-11-12 (twelfth session, step 4: development report):** three commits.
> - **Question round (user chose every recommendation; `m3_plan.md` step 4):**
>   - disclose the non-nesting, plus a reported-only *CQR, rearranged* row (also scored in step 6);
>   - four figures;
>   - M2's day-block bootstrap, plus a 7-day-block sensitivity for the pooled headline rows.
> - **`f806044` and `02681d6` (step 4a):** `src/gridcast/intervals/evaluation.py`, `report.py`, `gridcast report-m3` / `make report-m3`, and the unit and integration tests. The second commit adds a generated note on months with missing target days: the GFS temperature gap of Dec 2023 → Jan 2024 leaves 13 scored days in Jan 2024.
>   - The report recomputes every number from the stored bounds and stops unless the pooled values equal the stored metrics.
>   - It refuses runs with another interval config hash, and it runs the interval leakage tests itself (U4).
> - **Step 4b:** `make report-m3` ran from `02681d6` (about 1 min) and wrote [`reports/m3_intervals.md`](../../reports/m3_intervals.md) and `reports/figures/m3_*.png`. **U1–U4 pass on the development period.**
>   - U1: 79.40% at 80%.
>   - U2: every lead 78.97–80.36%.
>   - U3: 265 of 265 leads from 2 h; pooled IS80 1,248.1 vs 2,256.1 MW.
>   - U4: 7 passed.
>   - Reported only: CQR beats conformal-around-C on IS80 (−19.0 MW, CI excludes 0) but not on WIS (−0.1, CI includes 0). Coverage drops on hot-forecast days (71.5%) and public holidays (69.7%). Outcome-defined groups are lower still, as expected by construction.
>
> **Next:** step 5. Freeze the interval recipe first: the hash of the `intervals:` config plus the code commit, recorded in config and guarded by a unit test, before any AEMO data is compared. Then load `PDPASA_REGIONSOLUTION` (`LOR` runs, VIC1; about 1.1 GB) and write `reports/m3_aemo.md` (Oct 2024 → Sep 2025, §2.6). Questions for its round:
> - what exactly the freeze covers (the code paths and whether the rearranged row is part of the frozen recipe);
> - how to show the AEMO comparison at 30-minute resolution.
>
> **Update, 2026-11-11 (eleventh session, step 3: CQR):** two commits.
> - **`8168a39` (step 3a):** `src/gridcast/intervals/cqr_batch.py`, `gridcast intervals-cqr` / `make intervals-cqr`.
>   - It corrects the QR batch named by the new `intervals.qr_batch` with step 1's machinery: one score per side, the same 24 streams and the per-retrain selection, calibrating from 2022-11-01 on run 2's common mask.
>   - It refuses a mismatched QR batch (recipe hash, source batch, issue times, actuals, quantiles missing on scored pairs), and the CLI refuses a dirty tree.
>   - Levels run in parallel processes; a unit test checks the merge equals a single run.
>   - `intervals.qr_batch` is excluded from the interval config hash, so the hash stays `d34273400899af69`, pinned by a test.
> - **Step 3b:** the run took 160 s from `8168a39` and stored CQR batch **`484194d7-2e62-4176-80b0-b8f117354929`** (run 86), 383 MB. Log: `<data root>/logs/m3/intervals-cqr-20261111.log`.
>   - Pooled development results (reported only; step 4 computes U1–U4): cov50 49.7%, cov80 79.4%, cov95 93.9%, IS80 1248.1, WIS 150.5. For comparison, conformal-around-C (step 1) had 79.1%, 1267.2 and 150.6, and raw QR 64.7%, 1382.2 and 159.6.
>   - **To disclose in step 4:** 14.3% of scored pairs are not nested across levels, against 1.4% for conformal-around-C. Most of these pairs are between 80% and 95%. The per-level corrections cause it; each level's coverage and IS are unaffected, but the WIS and the reliability diagram combine levels. Details in `m3_plan.md` step 3.
>
> **Next:** step 4, the development report (`gridcast report-m3`, `reports/m3_intervals.md`; U1–U4 computed in code, leakage tests run by the report). Questions for its round:
> - how to show the non-nesting: a disclosure only, or also a rearranged CQR as a reported-only row;
> - which figures to include;
> - the bootstrap settings (M2's block bootstrap by day).
>
> **Update, 2026-11-11 (tenth session, step 2b: the QR development run):** `gridcast backtest-qr` ran overnight from a worktree at `6fdbac5` (01:23 → 09:10 AEDT; 28,059 s in the run) and stored QR batch **`4bab11d7-4301-4c74-9f32-ddd9644f3921`** (run 85, QR config hash `8767a8948966637c`). Log: `<data root>/logs/m3/backtest-qr-20261111-0123.log`.
> - Point reproduction passed in all 12 folds (48 refits; 14,436,480 pairs equal to run 2's).
> - Of 288 quantile selections, 201 chose `latest` and 228 chose 31 leaves; 110 matched the point model's setting. 23.4% of pairs crossed before sorting.
> - Raw QR, pooled over the 10,662,528 scored pairs (reported only): cov50 35.8%, cov80 64.7%, cov95 87.6%, IS80 1382.2, WIS 159.6. It under-covers on both sides, more often above. Details in `m3_plan.md` step 2.
> - Storage: 505 MB of quantiles under `<data root>/intervals/`, 265 MB of models under `<data root>/models/`. The worktree `../gridcast-au-qr` was removed afterwards.
>
> **Next:** step 3, CQR. Apply the step-1 conformal machinery (time-of-day grouping, per-side ACI, per-retrain selection) to the QR scores from batch `4bab11d7…`, and store the CQR batch.
>
> **Update, 2026-11-11 (tenth session, step 2a: QR code, smoke runs, a design change):**
> - **Code:** `src/gridcast/intervals/quantile.py` (the quantile model, per-quantile selection and the QR walk-forward) and `qr_batch.py` (`gridcast backtest-qr`, `make backtest-qr`). Placed under `intervals/` so the frozen M2 paths stay unchanged. Step 1's `batch.py` gained shared metric and storage helpers (`bounds_metrics`, `per_lead_rows`, `insert_run`); its outputs are unchanged.
>   - C trains on the rows M2's A, C and D share, as in run 2.
>   - At every (retrain, band) the loop also refits run 2's point setting with its stored tree count. The row counts must match run 2's records, and the fold forecasts must equal run 2's stored ones exactly; otherwise the run stops and nothing is stored.
>   - Tests: `tests/unit/test_quantile.py` (selection, exact reproduction and refusals, a leakage property with its control) and `tests/integration/test_intervals_qr.py`.
> - **Smoke run 1** (first retrain, reusing run 2's settings as planned): 203 s, point forecasts reproduced exactly. That projects 1.5 h for the full run, not the planned 4–6 h. Raw QR on the fold (pre-scoring): 80% coverage 67.2%, IS80 1339.4, 22.4% of pairs crossed before sorting.
> - **User decision 1:** since compute was the reason for reusing the point settings, switch to **per-quantile selection**. Each quantile scores M2's 8 settings (4 target forms × 31/127 leaves) by pinball loss on the early-stopping slice and refits the best.
> - **Smoke run 2** (per-quantile selection): 1,206 s, reproduced exactly, projecting **9.0–11.0 h**. On the same fold it was slightly worse: IS80 1370.5, WIS 155.9 against 154.1, coverage 68.4%, 27.5% crossed.
> - **User decision 2:** keep per-quantile selection and run it overnight, since one retrain is thin evidence. Both decisions and the smoke numbers are in `m3_plan.md` §1, §2.3, §5 and §6. Logs: `<data root>/logs/m3/backtest-qr-smoke-20261111.log` and `backtest-qr-smoke2-20261111.log`.
>
> **Next:** the full development run (`gridcast backtest-qr`, refuses a dirty tree), from a worktree at the step-2a commit with its own `.venv` and a copy of `.env`, as M2's holdout run did. Then record the outcome (step 2b) and go on to step 3 (CQR).
>
> **Update, 2026-11-10 (ninth session, step 1 done):** two commits.
> - **`819e862` (step 1a):** the interval core in `src/gridcast/intervals/` (scores; exact rolling order statistics; conformal with time-of-day grouping, per-side ACI and per-retrain selection; the CQR form, ready for step 3), with its tests.
>   - The tests: an exact match against a slow reference, and leakage property tests whose controls catch an off-by-one information rule.
>   - Also in this commit: migration 0005 (`model_kind = 'interval'`) and `gridcast intervals-conformal`.
>   - **Pre-scoring spike** ([`docs/m3_prescoring_spike.md`](../m3_prescoring_spike.md)). The user chose the recommended option each time: 8 × 3 h time-of-day blocks, ACI γ ∈ {0, 0.001, 0.005}, cold days at forecast daily maximum ≤ 12 °C. These were recorded in the plan and `configs/base.yaml` before any scored interval existed.
> - **Step 1b:** `gridcast intervals-conformal` ran from `819e862` in 308 s and stored interval batch **`72eb5209-9bef-4870-a375-131a86ee7e0f`**: interval config hash `d34273400899af69`, 1.4 GB under `<data root>/intervals/`, log `<data root>/logs/m3/intervals-conformal-20261110.log`.
>   - Pooled development results printed by the command (reported only; U1–U3 judge CQR, from step 3):
>
>     | run | cov50 % | cov80 % | cov95 % | IS80 | WIS |
>     |---|---|---|---|---|---|
>     | conformal_lgbm_C | 50.3 | 79.1 | 94.4 | 1267.2 | 150.6 |
>     | conformal_persistence | 49.4 | 80.0 | 94.0 | 2717.0 | 384.9 |
>     | conformal_daily_naive | 49.8 | 79.5 | 94.2 | 2407.3 | 281.7 |
>     | conformal_weekly_naive | 50.7 | 78.3 | 92.8 | 2752.1 | 322.5 |
>
>   - Every LightGBM C selection chose a per lead × time-of-day stream. At 80%, an ACI stream (γ > 0) won 23 of the 36 (retrain, band) selections, although ACI did not help on pre-scoring data. No interval was empty.
>
> **Next:** step 2, the quantile LightGBM walk-forward (development). Reuse run 2's per-retrain selections (plan §2.3), smoke-run the first retrain (2022-11-01) to measure the cost, and ask the user again before the full run if it looks longer than about 10 h.
>
> **Update, 2026-11-10 (eighth session, step 0 done):** the AEMO POE spike is in [`docs/m3_data_spike.md`](../m3_data_spike.md) (script `scripts/spikes/m3/aemo_poe.py`; four sampled months cached under `<data root>/raw/spikes/m3/nemweb/`).
> - `PDPASA_REGIONSOLUTION` (`LOR` runs) archives every 30-minute run with `DEMAND10/50/90`.
> - Its POE90–POE10 band is exactly symmetric and covered 66–78% of actual demand.
> - `DEMAND50` sits an unexplained ~+40 MW above the matched pre-dispatch `TOTALDEMAND`.
> - **User decision:** include the comparison, with the band as published as the headline and a band re-centred on pre-dispatch beside it; disclose the offset. The plan's §1, §2.6 and step 0 are updated.
>
> **Next:** step 1. Build the interval core (`src/gridcast/intervals/`: scores, rolling conformal, ACI, per-retrain selection, leakage tests). Then run the pre-scoring spike (issue times before 2023-08-01 only) that fixes the ACI step values, the time-of-day blocks and the cold threshold, and record them in the plan before any scored interval exists. Then `gridcast intervals-conformal`.
>
> **Update, 2026-11-10 (eighth session, M3 planned):** three question rounds with the user settled the M3 design, recorded in `m3_plan.md` §1 before any interval code or result existed:
> - **Methods:** conformal intervals around the stored forecasts, quantile LightGBM C, and CQR (the pre-registered headline).
> - **Scope:** demand only, at 80% (with 50% and 95% also reported).
> - **Settings:** conformal settings re-selected at every retrain from past data; QR reuses the point model's per-retrain choices.
> - **Extreme events:** grouped both by what was known at issue time and by outcome, reported only.
> - **Evaluation:** develop on issue times 1 Aug 2023 → 30 Sep 2025, freeze, then score the holdout once (U5). The holdout's point errors were already seen; this is disclosed.
> - **AEMO:** a step-0 spike decides whether AEMO's POE10/50/90 demand can be a benchmark.
>
> Success criteria U1–U5 are in `m3_plan.md` §3. Branch `m3-prediction-intervals`, from `main` at `a1b26bb`.
>
> **Next:** step 0 (the AEMO POE spike), then step 1 (interval core, conformal intervals, the pre-scoring spike that fixes the candidate grid).

## Local environment

Unchanged from M2 (see `m2_handoff.md`, *Local environment*):
- **Data:** under `GRIDCAST_DATA_ROOT=/Volumes/BIWIN/gridcast-au/data` on the external SSD. Mount it first.
- **Database:** PostgreSQL in the Docker volume `gridcast-au_pgdata` (`make db-up`).

Inputs M3 reads:

| Input | Where |
|---|---|
| Development point forecasts (run 2) | batch `main` = `6d783932-5834-4e18-a9e8-0e3567db46e6`, parquet under `<data root>/backtests/` |
| Run 2's per-retrain selections (for QR) | the `model_runs.config` records of that batch |
| Conformal intervals around C and the baselines (step 1) | batch `72eb5209-9bef-4870-a375-131a86ee7e0f`, parquet under `<data root>/intervals/` |
| Quantile LightGBM C, development (step 2) | batch `4bab11d7-4301-4c74-9f32-ddd9644f3921`, rearranged quantiles under `<data root>/intervals/`, models under `<data root>/models/` |
| CQR on quantile LightGBM C, development (step 3, headline) | batch `484194d7-2e62-4176-80b0-b8f117354929` (run 86), bounds under `<data root>/intervals/` |
| Development report (step 4) | `reports/m3_intervals.md`, regenerated by `make report-m3` (about 1 min) |
| Half-hour intervals, development (step 5) | batch `27aa78d6-8ab8-4509-aa40-d8b08cda68ea` (runs 87–91), bounds under `<data root>/intervals/` |
| Frozen interval recipe (step 5) | `intervals.frozen_config_hash` = `d34273400899af69`, `frozen_qr_config_hash` = `8767a8948966637c`, code at `bc5337e`; `make intervals-freeze-check` |
| AEMO POE band (step 5) | table `aemo_demand_poe` (migration 0006), PDPASA `LOR` runs Sep 2024 → Sep 2026; `reports/m3_aemo.md` via `make report-m3-aemo` (seconds) |
| Holdout point forecasts (step 6 only) | batch `holdout_main` = `fcb6abb6-49a3-4923-9d79-64e05a44d75f` |
| Frozen M2 recipe (must stay reproducible) | `backtest.frozen_ml_config_hash` = `b88eae0a04e176af`, code at `96a6e50` |

Sanity check before starting:

```bash
ls /Volumes/BIWIN/gridcast-au/data         # SSD mounted?
make db-up && uv run alembic current        # expect 0006 (head)
make lint typecheck test test-integration
```
