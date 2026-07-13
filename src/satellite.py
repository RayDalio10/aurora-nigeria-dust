"""Satellite dust observations.

Two references:
  - Sentinel-5P Absorbing Aerosol Index (AAI) via Google Earth Engine
    (used for dust-event indication).
  - MODIS MOD04_L2 aerosol optical depth via NASA Earthdata / earthaccess,
    using the Dark Target / Deep Blue Combined product (used for spatial and
    temporal validation). MOD04 files are HDF4 and must be read with pyhdf.
"""
from datetime import datetime, timedelta

import numpy as np
import ee
import earthaccess
from pyhdf.SD import SD, SDC

from config import EE_PROJECT

S5P_COLLECTION = "COPERNICUS/S5P/OFFL/L3_AER_AI"
MODIS_SHORT_NAME = "MOD04_L2"
NIGERIA_BBOX = (2, 3, 15, 16)  # (west, south, east, north)

# MODIS scientific dataset names
MODIS_AOD_DARKTARGET = "Optical_Depth_Land_And_Ocean"
MODIS_AOD_COMBINED = "AOD_550_Dark_Target_Deep_Blue_Combined"
MODIS_AOD_COMBINED_QA = "AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag"
MODIS_FILL = -9999
MODIS_SCALE = 0.001


# --- Earth Engine / Sentinel-5P AAI ---

def init_earth_engine():
    """Authenticate and initialize Earth Engine."""
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
    print("Earth Engine connected.")


def get_satellite_dust(date, scale=25000):
    """Fetch Sentinel-5P AAI over Nigeria for one date (YYYY-MM-DD).

    Returns cleaned (lons, lats, aai) arrays for the day's mean field.
    """
    next_day = (datetime.strptime(date, "%Y-%m-%d")
                + timedelta(days=1)).strftime("%Y-%m-%d")
    nigeria = ee.Geometry.Rectangle(list(NIGERIA_BBOX))
    image = (ee.ImageCollection(S5P_COLLECTION)
             .select("absorbing_aerosol_index")
             .filterDate(date, next_day)
             .filterBounds(nigeria)
             .mean()
             .clip(nigeria))

    data = (image.pixelLonLat().addBands(image)
            .reduceRegion(reducer=ee.Reducer.toList(), geometry=nigeria,
                          scale=scale, maxPixels=int(1e8))
            .getInfo())

    lons = np.array(data["longitude"], dtype=float)
    lats = np.array(data["latitude"], dtype=float)
    aai = np.array(data["absorbing_aerosol_index"], dtype=float)

    n = min(len(lons), len(lats), len(aai))
    lons, lats, aai = lons[:n], lats[:n], aai[:n]
    mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(aai)
    return lons[mask], lats[mask], aai[mask]


# --- MODIS AOD via earthaccess + pyhdf ---

def init_earthdata():
    """Authenticate to NASA Earthdata (interactive)."""
    earthaccess.login(strategy="interactive", persist=True)
    print("Earthdata connected.")


def _read_modis_granule(path, aod_var, qa_var=None, qa_min=None):
    """Read one MOD04 HDF4 granule; return (lon, lat, aod) flattened arrays.

    Applies the fill-value mask and scale factor. If qa_var and qa_min are
    given, keeps only pixels whose QA flag meets the minimum confidence.
    """
    hdf = SD(str(path), SDC.READ)
    try:
        aod = hdf.select(aod_var)[:].astype(float).ravel()
        lat = hdf.select("Latitude")[:].astype(float).ravel()
        lon = hdf.select("Longitude")[:].astype(float).ravel()
        aod = np.where(aod <= MODIS_FILL, np.nan, aod * MODIS_SCALE)
        if qa_var is not None and qa_min is not None:
            qa = hdf.select(qa_var)[:].astype(float).ravel()
            aod = np.where(qa >= qa_min, aod, np.nan)
    finally:
        hdf.end()
    return lon, lat, aod


def get_modis_aod(date, aod_var=MODIS_AOD_COMBINED, qa_var=None, qa_min=None,
                  local_path="modis_files"):
    """Fetch MODIS AOD over Nigeria for a single date (YYYY-MM-DD).

    Returns cleaned (lons, lats, aod). By default uses the Combined product;
    pass aod_var=MODIS_AOD_DARKTARGET for Dark Target only. Optional QA
    filtering keeps only confident pixels.
    """
    results = earthaccess.search_data(
        short_name=MODIS_SHORT_NAME, temporal=(date, date),
        bounding_box=NIGERIA_BBOX)
    print(f"Found {len(results)} MODIS granules for {date}")
    if not results:
        return None, None, None
    paths = earthaccess.download(results, local_path=local_path)
    return _collect_modis(paths, aod_var, qa_var, qa_min)


def get_modis_aod_composite(center_date, window_days=5, aod_var=MODIS_AOD_COMBINED,
                            qa_var=None, qa_min=None, local_path="modis_comp"):
    """Composite MODIS AOD over center_date +/- window_days to fill swath gaps.

    Returns cleaned (lons, lats, aod) pooled across all granules in the window.
    Temporal compositing is needed because single-day polar-orbiter coverage
    over the region is sparse.
    """
    d0 = datetime.strptime(center_date, "%Y-%m-%d")
    start = (d0 - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (d0 + timedelta(days=window_days)).strftime("%Y-%m-%d")
    results = earthaccess.search_data(
        short_name=MODIS_SHORT_NAME, temporal=(start, end),
        bounding_box=NIGERIA_BBOX)
    print(f"Found {len(results)} granules over {start}..{end}")
    if not results:
        return None, None, None
    paths = earthaccess.download(results, local_path=local_path)
    return _collect_modis(paths, aod_var, qa_var, qa_min)


def _collect_modis(paths, aod_var, qa_var, qa_min):
    """Read and pool a list of MOD04 granules, cropped to Nigeria."""
    all_lon, all_lat, all_aod = [], [], []
    for p in paths:
        try:
            lon, lat, aod = _read_modis_granule(p, aod_var, qa_var, qa_min)
            all_lon.append(lon)
            all_lat.append(lat)
            all_aod.append(aod)
        except Exception as exc:
            print("  skipped granule:", str(exc)[:80])
    if not all_aod:
        return None, None, None
    lon = np.concatenate(all_lon)
    lat = np.concatenate(all_lat)
    aod = np.concatenate(all_aod)
    w, s, e, n = NIGERIA_BBOX
    mask = ((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
            & np.isfinite(aod))
    return lon[mask], lat[mask], aod[mask]
