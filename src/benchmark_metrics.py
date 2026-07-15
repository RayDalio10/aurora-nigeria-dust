"""Aggregate the benchmark's per-date records into robust skill statistics.

Reads the JSON records saved by run_benchmark.py (via benchmark_store) and
computes, per lead time and across all dates:
  - Pearson and Spearman correlation between forecast and MODIS region-means,
    with bootstrap confidence intervals,
  - the same for the northern band vs the whole domain,
  - event detection (POD, FAR) aggregated over all dates,
  - coverage accounting (how many dates had valid MODIS).

CPU-only; no downloads. Everything derives from the saved records.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

import benchmark_store as store

LEADS = [1, 2, 3, 4, 5]


def _paired_series(records, lead, region="whole"):
    """Return matched (forecast, modis) arrays for one lead across all dates.

    region: 'whole' or 'north'. Drops dates with missing MODIS.
    """
    fc, ob = [], []
    fkey = "forecast_north" if region == "north" else "forecast_whole"
    for rec in records:
        leg = rec.get("leads", {}).get(str(lead))
        if not leg:
            continue
        f = leg.get(fkey)
        o = leg.get("modis_whole")
        if f is None or o is None:
            continue
        if np.isnan(f) or np.isnan(o):
            continue
        fc.append(f)
        ob.append(o)
    return np.array(fc, dtype=float), np.array(ob, dtype=float)


def _bootstrap_ci(x, y, stat="pearson", n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap confidence interval for a correlation statistic.

    Returns (point_estimate, low, high). Resamples paired points with
    replacement. Returns NaNs if fewer than 4 pairs.
    """
    n = len(x)
    if n < 4:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)

    def corr(a, b):
        if stat == "spearman":
            r, _ = spearmanr(a, b)
        else:
            r, _ = pearsonr(a, b)
        return r

    point = corr(x, y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots.append(corr(x[idx], y[idx]))
        except Exception:
            continue
    boots = np.array([b for b in boots if np.isfinite(b)])
    if boots.size == 0:
        return (point, float("nan"), float("nan"))
    low = float(np.percentile(boots, 100 * alpha / 2))
    high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (float(point), low, high)


def skill_by_lead(records, region="whole"):
    """Per-lead Pearson and Spearman with bootstrap CIs, across all dates."""
    out = {}
    for lead in LEADS:
        x, y = _paired_series(records, lead, region)
        p, plo, phi = _bootstrap_ci(x, y, "pearson")
        s, slo, shi = _bootstrap_ci(x, y, "spearman")
        out[lead] = {
            "n": int(len(x)),
            "pearson": p, "pearson_ci": [plo, phi],
            "spearman": s, "spearman_ci": [slo, shi],
        }
    return out


def event_detection(records, lead=2, percentile=70, region="whole"):
    """Aggregate POD and FAR across all dates for one lead.

    An 'event' is a date whose observed region-mean is above the given
    percentile of all observed values; a 'warning' is above the same
    percentile of forecasts. Aggregating over all dates fixes the tiny-sample
    fragility of the single-case study.
    """
    x, y = _paired_series(records, lead, region)  # x=forecast, y=modis
    if len(x) < 4:
        return {"n": int(len(x)), "pod": float("nan"), "far": float("nan")}
    obs_event = y >= np.percentile(y, percentile)
    fc_warn = x >= np.percentile(x, percentile)
    hits = int(np.sum(obs_event & fc_warn))
    misses = int(np.sum(obs_event & ~fc_warn))
    false_alarms = int(np.sum(~obs_event & fc_warn))
    pod = hits / (hits + misses) if (hits + misses) else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else float("nan")
    return {"n": int(len(x)), "hits": hits, "misses": misses,
            "false_alarms": false_alarms, "pod": pod, "far": far}


def coverage_summary(records):
    """How many dates, and how many had valid MODIS at each lead."""
    total = len(records)
    per_lead = {}
    for lead in LEADS:
        valid = 0
        for rec in records:
            leg = rec.get("leads", {}).get(str(lead))
            if leg and not leg.get("modis_missing", True):
                valid += 1
        per_lead[lead] = valid
    return {"total_dates": total, "valid_modis_per_lead": per_lead}
