"""Global SHAP summaries of a stored sample (m4_plan.md §2.6; §1 "Clarified in step 5").

Every summary uses the **whole-forecast** decomposition
(:func:`~gridcast.explainability.global_run.whole_forecast`), except the table by target
form, which shows the raw tree-only values, and M2's gain importance, which is tree-only by
nature. A band's fits are pooled with one weight per pair. A family's contribution is the
sum of its features' contributions; its mean |SHAP| is the mean of the absolute family sum.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gridcast.config import ShapReportConfig
from gridcast.explainability.families import Families, families
from gridcast.explainability.global_run import StoredSample, shap_col, whole_forecast, x_col
from gridcast.timeutils import INTERVAL

WEATHER_FAMILY = "weather"
WEATHER_X = "wx_temp"


def band_label(band: tuple[int, int]) -> str:
    """Lead band in minutes or hours, e.g. (73, 288) → '6–24 h'."""
    lo, hi = (band[0] - 1) * 5, band[1] * 5
    if hi <= 60:
        return f"{band[0] * 5}–{hi} min"
    if lo < 60:
        return f"{band[0] * 5} min–{hi / 60:g} h"
    return f"{lo / 60:g}–{hi / 60:g} h"


@dataclass(slots=True)
class ShapSummary:
    bands: list[tuple[int, int]]
    families: Families
    fits: pd.DataFrame  # one row per fit: key, retrain, band, form, pairs, eligible, trees
    feature_abs: pd.DataFrame  # mean |SHAP| (whole forecast), feature × band
    family_abs: pd.DataFrame  # family × band
    family_abs_by_form: pd.DataFrame  # raw tree-only: rows (band, form), family columns
    gain_feature: pd.DataFrame  # mean gain share, feature × band
    gain_family: pd.DataFrame
    stability: pd.DataFrame  # band, retrain_a, retrain_b, form_a, form_b, spearman
    weather: pd.DataFrame  # band, lower, n, mean, lo, hi
    question5: pd.DataFrame  # basis, group, threshold, n_all, n_top, term, top, all, diff
    gate: dict
    pairs_by_band: dict[tuple[int, int], int]


def _band_key(band: tuple[int, int]) -> str:
    return band_label(band)


def summarise(sample: StoredSample, cfg: ShapReportConfig, tz: str) -> ShapSummary:
    columns = sample.columns
    fam = families(sample.manifest["variant"])
    if list(fam.columns) != columns:
        raise ValueError("the stored columns are not the variant's")
    n_fam = len(fam.keys)
    fits = sample.fits
    bands = sorted({tuple(f["band"]) for f in fits})

    abs_feat: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(len(columns)))
    abs_fam: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_fam))
    raw_fam: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_fam))
    raw_n: dict[tuple, int] = defaultdict(int)
    raw_fits: dict[tuple, int] = defaultdict(int)
    n_band: dict[tuple, int] = defaultdict(int)
    per_fit_rank: dict[tuple, list[tuple[str, str, np.ndarray]]] = defaultdict(list)
    gains: dict[tuple, list[np.ndarray]] = defaultdict(list)
    weather_parts: list[pd.DataFrame] = []
    q5_parts: list[pd.DataFrame] = []
    rows = []
    q5_band = tuple(cfg.question5_band)
    wcols = [j for j, c in enumerate(columns) if fam.of(c) == WEATHER_FAMILY]

    for f in sorted(fits, key=lambda f: (tuple(f["band"]), f["retrain"])):
        band, form = tuple(f["band"]), f["target_form"]
        frame = sample.frame(f["key"])
        whole = whole_forecast(frame, form, columns)
        fam_whole = fam.sums(whole.contributions)
        raw = frame[[shap_col(c) for c in columns]].to_numpy(dtype=np.float64)
        n = len(frame)
        abs_feat[band] += np.abs(whole.contributions).sum(axis=0)
        abs_fam[band] += np.abs(fam_whole).sum(axis=0)
        raw_fam[(band, form)] += np.abs(fam.sums(raw)).sum(axis=0)
        raw_n[(band, form)] += n
        raw_fits[(band, form)] += 1
        n_band[band] += n
        per_fit_rank[band].append((f["retrain"], form, np.abs(whole.contributions).mean(axis=0)))
        g = np.array([f["gain"][c] for c in columns], dtype=np.float64)
        gains[band].append(g / g.sum())
        weather_parts.append(
            pd.DataFrame(
                {
                    "band_start": band[0],
                    "band_end": band[1],
                    "temp": frame[x_col(WEATHER_X)].to_numpy(dtype=np.float64),
                    "contribution": whole.contributions[:, wcols].sum(axis=1),
                }
            )
        )
        if band == q5_band:
            part = pd.DataFrame(fam_whole, columns=list(fam.keys))
            part.insert(0, "constant", whole.constant)
            part["actual"] = frame["actual"].to_numpy(dtype=np.float64)
            part["forecast"] = frame["stored"].to_numpy(dtype=np.float64)
            starts = pd.DatetimeIndex(frame["target_end"]) - INTERVAL
            part["month"] = np.asarray(starts.tz_convert(tz).month)
            q5_parts.append(part)
        rows.append(
            {
                "key": f["key"],
                "retrain": f["retrain"],
                "band": band_label(band),
                "band_start": band[0],
                "form": form,
                "pairs": n,
                "eligible": f["eligible_pairs"],
                "trees": f["n_trees"],
                "leaves": f["num_leaves"],
            }
        )

    labels = [_band_key(b) for b in bands]
    feature_abs = pd.DataFrame(
        {_band_key(b): abs_feat[b] / n_band[b] for b in bands}, index=columns
    )[labels]
    family_abs = pd.DataFrame(
        {_band_key(b): abs_fam[b] / n_band[b] for b in bands}, index=list(fam.keys)
    )[labels]
    by_form = pd.DataFrame(
        [
            {
                "band": _band_key(b),
                "form": form,
                "fits": raw_fits[(b, form)],
                "pairs": raw_n[(b, form)],
                **dict(zip(fam.keys, raw_fam[(b, form)] / raw_n[(b, form)], strict=True)),
            }
            for (b, form) in sorted(raw_fam)
        ]
    )
    gain_feature = pd.DataFrame(
        {_band_key(b): np.mean(gains[b], axis=0) for b in bands}, index=columns
    )[labels]
    gain_family = pd.DataFrame(
        {_band_key(b): fam.sums(np.mean(gains[b], axis=0)[None, :])[0] for b in bands},
        index=list(fam.keys),
    )[labels]

    stab = []
    for b in bands:
        ranked = per_fit_rank[b]
        for (ra, fa, va), (rb, fb, vb) in itertools.pairwise(ranked):
            rho = float(spearmanr(va, vb).statistic)
            stab.append(
                {
                    "band": _band_key(b),
                    "retrain_a": ra,
                    "retrain_b": rb,
                    "form_a": fa,
                    "form_b": fb,
                    "spearman": rho,
                }
            )

    weather = _weather_bins(pd.concat(weather_parts, ignore_index=True), cfg)
    question5 = _question5(pd.concat(q5_parts, ignore_index=True), list(fam.keys), cfg)
    return ShapSummary(
        bands=bands,
        families=fam,
        fits=pd.DataFrame(rows),
        feature_abs=feature_abs,
        family_abs=family_abs,
        family_abs_by_form=by_form,
        gain_feature=gain_feature,
        gain_family=gain_family,
        stability=pd.DataFrame(stab),
        weather=weather,
        question5=question5,
        gate=dict(sample.manifest["gate"]),
        pairs_by_band={b: n_band[b] for b in bands},
    )


def _weather_bins(frame: pd.DataFrame, cfg: ShapReportConfig) -> pd.DataFrame:
    """Per band and fixed temperature bin: pairs, mean and the configured quantile range."""
    lo_q, hi_q = cfg.weather_range
    frame = frame.assign(lower=np.floor(frame["temp"] / cfg.weather_bin_c) * cfg.weather_bin_c)
    out = []
    for key, grp in frame.groupby(["band_start", "band_end", "lower"]):
        start, end, lower = (float(k) for k in np.asarray(key, dtype=np.float64))
        c = grp["contribution"].to_numpy()
        out.append(
            {
                "band": _band_key((int(start), int(end))),
                "band_start": int(start),
                "lower": float(lower),
                "n": len(c),
                "mean": float(c.mean()),
                "lo": float(np.quantile(c, lo_q)),
                "hi": float(np.quantile(c, hi_q)),
                "drawn": len(c) >= cfg.weather_min_pairs,
            }
        )
    return pd.DataFrame(out).sort_values(["band_start", "lower"]).reset_index(drop=True)


def _question5(frame: pd.DataFrame, family_keys: list[str], cfg: ShapReportConfig) -> pd.DataFrame:
    """Mean contributions in each group's top share against all of the group's targets."""
    terms = ["constant", *family_keys]
    groups: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(frame), dtype=bool))]
    groups += [
        (name, frame["month"].isin(months).to_numpy()) for name, months in cfg.seasons.items()
    ]
    out = []
    for basis in ("actual", "forecast"):
        for name, member in groups:
            sub = frame[member]
            if sub.empty:
                raise ValueError(f"question 5: group {name!r} has no targets in the sample")
            values = sub[basis].to_numpy()
            threshold = float(np.quantile(values, 1 - cfg.top_share))
            top = values > threshold
            for term in [*terms, "forecast_total"]:
                col = (
                    sub[terms].sum(axis=1).to_numpy()
                    if term == "forecast_total"
                    else sub[term].to_numpy()
                )
                out.append(
                    {
                        "basis": basis,
                        "group": name,
                        "threshold": threshold,
                        "n_all": len(sub),
                        "n_top": int(top.sum()),
                        "term": term,
                        "top": float(col[top].mean()),
                        "all": float(col.mean()),
                        "diff": float(col[top].mean() - col.mean()),
                    }
                )
    return pd.DataFrame(out)
