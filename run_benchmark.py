"""Resumable pilot benchmark runner (WO-6).

Loops the pilot initialization dates (2 Harmattan seasons, weekly sampling),
skips dates already saved, and for each remaining date:
  - runs the Aurora forecast (5-day rollout),
  - records region-mean PM10 at leads 1-5 (whole domain and northern band),
  - fetches the matched MODIS region-mean (windowed composite) for each lead's
    valid date,
  - saves one small JSON record to results/.

Stop any time (session reset, quota); re-run to resume. Commit the results/
folder to GitHub so progress is durable.

Prerequisites (once per session): CDSAPI_URL/CDSAPI_KEY env vars; Earth Engine
and Earthdata initialized; GPU available.

Run:  python run_benchmark.py
"""
from datetime import datetime, timedelta

import numpy as np

import config
import credentials
import forecast
import satellite
import benchmark_store as store

# --- Pilot configuration ---
# Two Harmattan seasons, weekly sampling across Dec-Jan-Feb.
SEASONS = [("2023-12-01", "2024-02-29"), ("2024-12-01", "2025-02-28")]
SAMPLE_EVERY_DAYS = 7
LEADS = [1, 2, 3, 4, 5]
NORTH_LAT_MIN = 12.0  # northern band for north-separate reporting (REQ-BENCH-002.2)


def pilot_dates():
    """Generate the weekly initialization dates across the pilot seasons."""
    dates = []
    for start, end in SEASONS:
        d = datetime.strptime(start, "%Y-%m-%d")
        end_d = datetime.strptime(end, "%Y-%m-%d")
        while d <= end_d:
            dates.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=SAMPLE_EVERY_DAYS)
    return dates


def region_means(preds, atm, step):
    """Return (whole_domain_mean, north_band_mean) of PM10 at a rollout step."""
    pm = preds[step].surf_vars["pm10"][0, 0].numpy()
    lats = atm.latitude.values
    whole = float(np.nanmean(pm))
    north_rows = lats >= NORTH_LAT_MIN
    north = float(np.nanmean(pm[north_rows, :])) if north_rows.any() else float("nan")
    return whole, north


_modis_cache = {}


def modis_mean(date):
    """MODIS Combined AOD region-mean for a date (+/- 1-day windowed composite)."""
    if date not in _modis_cache:
        try:
            lo, la, ao = satellite.get_modis_aod_composite(
                date, window_days=1,
                aod_var=satellite.MODIS_AOD_COMBINED,
                qa_var=satellite.MODIS_AOD_COMBINED_QA, qa_min=1)
            _modis_cache[date] = (
                float(np.nanmean(ao)) if ao is not None and len(ao) else float("nan"))
        except Exception:
            _modis_cache[date] = float("nan")
    return _modis_cache[date]


def process_date(init_date):
    """Run one initialization date and return its result record."""
    preds, atm = forecast.run_aurora_forecast(init_date, steps=10)
    record = {"leads": {}}
    for lead in LEADS:
        step = lead * 2 - 1
        whole, north = region_means(preds, atm, step)
        valid_date = (datetime.strptime(init_date, "%Y-%m-%d")
                      + timedelta(days=lead)).strftime("%Y-%m-%d")
        obs = modis_mean(valid_date)
        record["leads"][str(lead)] = {
            "valid_date": valid_date,
            "forecast_whole": whole,
            "forecast_north": north,
            "modis_whole": obs,
            "modis_missing": bool(np.isnan(obs)),
        }
    return record


def main():
    credentials.write_cdsapirc_from_env()
    satellite.init_earth_engine()
    satellite.init_earthdata()

    dates = pilot_dates()
    done = set(store.list_completed_dates())
    todo = [d for d in dates if d not in done]
    print(f"Pilot dates: {len(dates)} total, {len(done)} done, {len(todo)} to do.")

    for i, d in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {d} ...")
        try:
            record = process_date(d)
            store.save_result(d, record)
            print(f"  saved {d}")
        except Exception as exc:
            print(f"  FAILED {d}: {str(exc)[:120]} (will retry next run)")

    print("Chunk complete. Commit the results/ folder to GitHub to preserve progress.")


if __name__ == "__main__":
    main()
  
