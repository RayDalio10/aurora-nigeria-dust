"""Reproduce the full case study end to end and save every figure.

This single script regenerates all the main project figures cleanly (each is
saved to figures/ via savefig, never display-only):
  1. 5-day forecast panels for 2024 and 2026
  2. Three-year Day-3 comparison (2024 / 2025 / 2026)
  3. Sentinel-5P AAI satellite map (15 Jan 2024)
  4. Forecast-vs-observation side-by-side (regridded) for 15 Jan 2024
  5. CAMS-vs-MODIS spatial comparison map + raw/log scatter (24 Jan 2024 window)
  6. Skill-vs-lead-time curve (the headline result)
  7. Event-detection scores (printed)

Prerequisites (once per environment):
  - Copernicus ADS credentials: set CDSAPI_URL and CDSAPI_KEY env vars.
  - Earthdata: interactive login on first run.
  - Earth Engine: interactive init on first run.
  - GPU required for Aurora inference.

Run:  python run_case_study.py
"""
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import config
import credentials
import forecast
import satellite
import validation as val
import plots

CASE_DATE = "2024-01-15"
DUST_DAY = "2024-01-24"


def region_mean_forecast(preds, step):
    """Region-mean PM10 for a given rollout step."""
    return float(np.nanmean(preds[step].surf_vars["pm10"][0, 0].numpy()))


def main():
    credentials.write_cdsapirc_from_env()
    satellite.init_earth_engine()
    satellite.init_earthdata()

    # ---- 1. Single forecasts + 5-day evolution panels (2024 and 2026) ----
    preds24, atm24 = forecast.run_aurora_forecast("2024-01-15", steps=10)
    lon24, lat24 = atm24.longitude.values, atm24.latitude.values
    plots.plot_forecast_panels(preds24, lon24, lat24, "15 Jan 2024",
                               "forecast_5day_2024.png")

    preds26, atm26 = forecast.run_aurora_forecast("2026-01-15", steps=10)
    lon26, lat26 = atm26.longitude.values, atm26.latitude.values
    plots.plot_forecast_panels(preds26, lon26, lat26, "15 Jan 2026",
                               "forecast_5day_2026.png")

    # Single 2-day spike map (Day 1 step) for 2024
    pm10_day1 = preds24[1].surf_vars["pm10"][0, 0].numpy()
    plots.plot_dust_map(lon24, lat24, pm10_day1,
                        "Aurora PM10 dust forecast over Nigeria (2 days ahead, 15 Jan 2024)",
                        "forecast_single_2024.png")

    # ---- 2. Three-year Day-3 comparison (2024 / 2025 / 2026) ----
    preds25, atm25 = forecast.run_aurora_forecast("2025-01-15", steps=10)
    plots.plot_three_year_day3(
        [("2024-01-15", preds24, atm24),
         ("2025-01-15", preds25, atm25),
         ("2026-01-15", preds26, atm26)],
        "three_year_day3.png")

    # ---- 3. Sentinel-5P AAI satellite map (15 Jan 2024) ----
    a_lons, a_lats, aai = satellite.get_satellite_dust(CASE_DATE)
    plots.plot_scatter_map(a_lons, a_lats, aai,
                           "Sentinel-5P AAI over Nigeria (15 Jan 2024)",
                           "aai_map_2024.png", cbar_label="Absorbing Aerosol Index")

    # ---- 4. Forecast vs observation side-by-side (regridded), 15 Jan 2024 ----
    aurora_grid = val.regrid_gridded(lon24, lat24, pm10_day1,
                                     grid_lon=config.GRID_LON, grid_lat=config.GRID_LAT,
                                     cell=0.5)
    sat_grid = val.regrid_points(a_lons, a_lats, aai,
                                 grid_lon=config.GRID_LON, grid_lat=config.GRID_LAT,
                                 cell=0.5)
    plots.plot_two_maps(aurora_grid, sat_grid,
                        "Aurora PM10 (regridded)", "Sentinel-5P AAI (regridded)",
                        "forecast_vs_obs_2024.png")

    # ---- 5. CAMS dust-AOD vs MODIS AOD (24 Jan window), map + scatter ----
    # Both are temporal composites over the same +/- 15-day window, matched in
    # time and coverage. This is the like-for-like AOD-vs-AOD comparison.
    forecast.download_cams_dustaod("2024-01-09/2024-02-08", "cams_dustaod_window.zip")
    c_lons, c_lats, c_field = forecast.load_cams_dustaod(
        "cams_dustaod_window.zip", "cams_dustaod_window")
    cams_grid = val.regrid_gridded(c_lons, c_lats, c_field, cell=1.0, min_points=1)

    cb_lons, cb_lats, cb_aod = satellite.get_modis_aod_composite(
        DUST_DAY, window_days=15,
        aod_var=satellite.MODIS_AOD_COMBINED,
        qa_var=satellite.MODIS_AOD_COMBINED_QA, qa_min=1)
    modis_grid = val.regrid_points(cb_lons, cb_lats, cb_aod, cell=1.0, min_points=3)

    # Side-by-side composite comparison map
    plots.plot_two_maps(cams_grid, modis_grid,
                        "CAMS composite dust-AOD", "MODIS composite AOD",
                        "cams_vs_modis_2024.png")

    # Spatial correlation (log-space, since AOD is log-normal) + scatter
    r_log, n_log = val.spatial_correlation(cams_grid, modis_grid, log=True)
    r_spear, _ = val.spatial_correlation(cams_grid, modis_grid, method="spearman")
    print(f"CAMS vs MODIS spatial: log-Pearson {r_log}, Spearman {r_spear}, n={n_log}")

    a = cams_grid.ravel()
    b = modis_grid.ravel()
    both = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    plots.plot_scatter(a[both], b[both], "CAMS dust-AOD", "MODIS Combined AOD",
                       "cams_vs_modis_scatter_raw.png", title="Raw AOD")
    plots.plot_scatter(a[both], b[both], "log(CAMS)", "log(MODIS)",
                       "cams_vs_modis_scatter_log.png", title="Log AOD", log=True)

    # ---- 6. Skill-vs-lead-time curve (headline) ----
    start_days = [(datetime(2024, 1, 7) + timedelta(days=2 * i)).strftime("%Y-%m-%d")
                  for i in range(10)]
    forecast_by_lead = {lead: [] for lead in (1, 2, 3, 4, 5)}
    for sd in start_days:
        p, _ = forecast.run_aurora_forecast(sd, steps=10)
        for lead in (1, 2, 3, 4, 5):
            step = lead * 2 - 1
            vdate = (datetime.strptime(sd, "%Y-%m-%d")
                     + timedelta(days=lead)).strftime("%Y-%m-%d")
            forecast_by_lead[lead].append((vdate, region_mean_forecast(p, step)))

    cache = {}
    def modis_mean(d):
        # Use a +/- 1-day MODIS composite (3 days of passes) rather than a
        # single exact day. Single-day polar-orbiter coverage over Nigeria is
        # too sparse and gappy, which injects noise into the region-average and
        # collapses the correlation. The windowed composite matches the method
        # used to produce the documented skill curve. Note: this is a
        # methodological choice with a real effect on the numbers; the exact
        # values remain preliminary (n=10) pending the multi-season benchmark.
        if d not in cache:
            lo, la, ao = satellite.get_modis_aod_composite(
                d, window_days=1,
                aod_var=satellite.MODIS_AOD_COMBINED,
                qa_var=satellite.MODIS_AOD_COMBINED_QA, qa_min=1)
            cache[d] = float(np.nanmean(ao)) if ao is not None and len(ao) else np.nan
        return cache[d]

    skill = val.skill_vs_lead_time(forecast_by_lead, modis_mean)
    leads = [1, 2, 3, 4, 5]
    pear = [skill[l]["pearson"] for l in leads]
    spear = [skill[l]["spearman"] for l in leads]
    for l in leads:
        print(f"Lead {l}d: Pearson {skill[l]['pearson']}, "
              f"Spearman {skill[l]['spearman']}, n={skill[l]['n']}")
    plots.plot_skill_vs_leadtime(leads, pear, spear)

    # ---- 7. Event detection at 2-day lead ----
    pairs = forecast_by_lead[2]
    fc = np.array([v for (_, v) in pairs])
    ob = np.array([modis_mean(d) for (d, _) in pairs])
    ev = val.event_detection(fc, ob, percentile=70)
    print("Event detection (2-day lead):", ev)

    print("Done. All figures saved in figures/.")


if __name__ == "__main__":
    main()
