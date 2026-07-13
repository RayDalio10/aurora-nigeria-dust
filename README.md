# Aurora Nigeria Dust — Harmattan Dust Forecasting and Validation

Evaluating whether the Aurora Earth-system foundation model can forecast
Harmattan dust over Nigeria with enough skill to support early warning.

## Summary

Using the pre-trained Aurora 0.4-degree air-pollution model (inference only),
this project forecasts dust (PM10) over Nigeria from CAMS input data and
validates the forecasts against independent satellite observations
(MODIS aerosol optical depth; Sentinel-5P Absorbing Aerosol Index).

Preliminary result (single-season case study, January 2024): Aurora's
region-averaged dust forecast correlates with MODIS observations by lead time
at approximately Day 1 r=0.93, Day 2 r=0.81, Day 3 r=0.48, Day 4 r=0.11,
Day 5 r=-0.30 (n=10) - a useful early-warning horizon of about 1-3 days.
These results are preliminary and to be confirmed across multiple seasons.

## Data sources

- CAMS global atmospheric-composition forecasts (Copernicus ADS) - model input and dust-AOD reference.
- Aurora checkpoint aurora-0.4-air-pollution.ckpt (Hugging Face microsoft/aurora). Weights are CC-BY-NC-SA (non-commercial).
- MODIS MOD04_L2 AOD (NASA Earthdata) - Dark Target / Deep Blue Combined product.
- Sentinel-5P Absorbing Aerosol Index (Google Earth Engine).

## Setup

1. Install dependencies: `pip install -r requirements.txt`.
2. Copernicus ADS: set env vars `CDSAPI_URL` and `CDSAPI_KEY` (get a free key from the ADS site).
3. NASA Earthdata: register at urs.earthdata.nasa.gov; first run calls an interactive login.
4. Google Earth Engine: register a project; set EE_PROJECT in src/config.py.
5. A GPU is required for Aurora inference (the region is cropped to Nigeria to fit free-tier GPUs).

## Usage
