"""Validation machinery.

Regrids forecasts and observations onto a shared grid and computes:
  - spatial correlation (Pearson/Spearman, raw or log) between two maps,
  - temporal correlation of region-averaged daily series,
  - forecast skill as a function of lead time,
  - event detection scores (probability of detection, false-alarm ratio).

AOD is approximately log-normal, so log-space correlation is provided.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from config import GRID_LON, GRID_LAT

# Coarser 1-degree grid used for the robust validation
GRID_LON_C = np.arange(2, 15, 1.0)
GRID_LAT_C = np.arange(3, 16, 1.0)


def regrid_points(lons, lats, values, grid_lon=GRID_LON_C, grid_lat=GRID_LAT_C,
                  cell=1.0, min_points=1):
    """Bin scattered (lon, lat, value) points onto a grid, binning by true
    coordinate value. Boxes with fewer than min_points valid points are NaN.

    Binning by coordinate keeps orientation correct regardless of how the
    source latitude is ordered.
    """
    grid = np.full((len(grid_lat), len(grid_lon)), np.nan)
    for i, la in enumerate(grid_lat):
        for j, lo in enumerate(grid_lon):
            in_box = ((lons >= lo) & (lons < lo + cell) &
                      (lats >= la) & (lats < la + cell))
            if np.sum(in_box) >= min_points:
                grid[i, j] = np.nanmean(values[in_box])
    return grid


def regrid_gridded(field_lons, field_lats, field_values, **kwargs):
    """Regrid an already-gridded field (e.g. Aurora or CAMS) onto the grid.

    Note: sort a descending-latitude source to ascending before calling, so
    the field aligns with the ascending grid convention.
    """
    lon_mesh, lat_mesh = np.meshgrid(field_lons, field_lats)
    return regrid_points(lon_mesh.ravel(), lat_mesh.ravel(),
                         np.asarray(field_values).ravel(), **kwargs)


def spatial_correlation(grid_a, grid_b, method="pearson", log=False):
    """Correlation between two regridded maps, over boxes valid in both.

    method: 'pearson' or 'spearman'. log: correlate log-values (both must be
    positive; use for log-normal AOD). Returns (r, n_boxes); r is None if
    fewer than three overlapping valid boxes.
    """
    a = grid_a.ravel()
    b = grid_b.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if log:
        mask &= (a > 0) & (b > 0)
    if mask.sum() < 3:
        return None, 0
    x, y = a[mask], b[mask]
    if log:
        x, y = np.log(x), np.log(y)
    if method == "spearman":
        r, _ = spearmanr(x, y)
    else:
        r, _ = pearsonr(x, y)
    return r, int(mask.sum())


def temporal_correlation(series_a, series_b):
    """Correlate two daily time series over days valid in both.

    Returns dict with Pearson, Spearman, and the number of paired days.
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return {"pearson": None, "spearman": None, "n": int(mask.sum())}
    pear, _ = pearsonr(a[mask], b[mask])
    spear, _ = spearmanr(a[mask], b[mask])
    return {"pearson": pear, "spearman": spear, "n": int(mask.sum())}


def skill_vs_lead_time(forecast_by_lead, observed_mean, leads=(1, 2, 3, 4, 5)):
    """Compute forecast skill at each lead time.

    forecast_by_lead: dict {lead: [(valid_date, forecast_value), ...]}.
    observed_mean: callable date -> observed region-mean (e.g. MODIS).
    Returns dict {lead: {'pearson':.., 'spearman':.., 'n':..}}.
    """
    out = {}
    for lead in leads:
        pairs = forecast_by_lead[lead]
        fc = np.array([v for (_, v) in pairs], dtype=float)
        ob = np.array([observed_mean(d) for (d, _) in pairs], dtype=float)
        out[lead] = temporal_correlation(fc, ob)
    return out


def event_detection(forecast_values, observed_values, percentile=70):
    """Event-detection scores at a given lead.

    An 'event' is a day above the given percentile of the observed series;
    a 'warning' is a day above the same percentile of the forecast series.
    Returns dict with the contingency counts, POD, and FAR.
    """
    fc = np.asarray(forecast_values, dtype=float)
    ob = np.asarray(observed_values, dtype=float)
    mask = np.isfinite(fc) & np.isfinite(ob)
    fc, ob = fc[mask], ob[mask]

    obs_event = ob >= np.percentile(ob, percentile)
    fc_warn = fc >= np.percentile(fc, percentile)

    hits = int(np.sum(obs_event & fc_warn))
    misses = int(np.sum(obs_event & ~fc_warn))
    false_alarms = int(np.sum(~obs_event & fc_warn))
    correct_neg = int(np.sum(~obs_event & ~fc_warn))

    pod = hits / (hits + misses) if (hits + misses) else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else np.nan
    return {
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "correct_negatives": correct_neg, "pod": pod, "far": far,
        "n": len(fc),
    }
