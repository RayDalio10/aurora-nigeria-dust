# Aurora Nigeria Dust — Harmattan Dust Forecasting and Validation

Evaluating whether the Aurora Earth-system foundation model can forecast
Harmattan dust over Nigeria with enough skill to support early warning.

## Summary

Using the pre-trained Aurora 0.4-degree air-pollution model (inference only),
this project forecasts dust (PM10) over Nigeria from CAMS input data and
validates the forecasts against independent satellite observations
(MODIS aerosol optical depth; Sentinel-5P Absorbing Aerosol Index).

Multi-season benchmark result (two Harmattan seasons, 2023/24 and 2024/25; 26 forecast dates): Aurora shows statistically significant dust-forecast skill
at 1-2 day lead - Day 1 Pearson r = 0.79 (95% CI 0.42-0.95), Day 2 r = 0.55 (95% CI 0.15-0.84) - with skill 
not distinguishable from zero beyond about two days. At a 2-day lead the system detects 75% of dust events with 
a 25% false-alarm ratio. This defines a useful early-warning horizon of about 1-2 days. Results are from a pilot benchmark (n = 26)
with wide confidence intervals that would tighten with a larger multi-season sample.

A single-season case study (January 2024, n = 10) gave a higher but sample-fragile Day-1 correlation (~0.93); the multi-season benchmark above is
the robust, defensible result.

## Data sources

- CAMS global atmospheric-composition forecasts (Copernicus ADS) - model input and dust-AOD reference.
- Aurora checkpoint aurora-0.4-air-pollution.ckpt (Hugging Face microsoft/aurora). Weights are CC-BY-NC-SA (non-commercial).
- MODIS MOD04_L2 AOD (NASA Earthdata) - Dark Target / Deep Blue Combined product.
- Sentinel-5P Absorbing Aerosol Index (Google Earth Engine).

## Setup

This project requires Python, a GPU-enabled runtime for Aurora inference, and access to three open data services: Copernicus ADS, NASA Earthdata, and Google Earth Engine.

### 1. Clone the repository
bash

git clone https://github.com/RayDalio10/aurora-nigeria-dust.git

cd aurora-nigeria-dust

### 2. Install dependencies
bash
pip install -r requirements.txt

### 3. Configure credentials

Create the required access accounts:

- Copernicus ADS for CAMS atmospheric data
- NASA Earthdata for MODIS AOD
- Google Earth Engine for Sentinel-5P AAI

Before running the code in a notebook or script, set the required environment variables:

python

import os

import getpass

os.environ["CDSAPI_URL"] = "https://ads.atmosphere.copernicus.eu/api"

os.environ["CDSAPI_KEY"] = getpass.getpass("Copernicus ADS key: ")

os.environ["EE_PROJECT"] = getpass.getpass("Google Earth Engine project ID: ")

Earthdata and Earth Engine will prompt for interactive authentication on first use.

### 4. Import the project modules

If running from a notebook, add the `src/` directory to the Python path:

python

import sys

sys.path.append("src")

import config

import credentials

import forecast

import satellite

import validation as val

import plots

import benchmark_store

import benchmark_metrics

credentials.write_cdsapirc_from_env()



Do not commit API keys, passwords, `.env` files, `.cdsapirc` files, downloaded datasets, model checkpoints, or raw satellite/CAMS files.

## Usage

### Run the January 2024 case study

bash

  python run_case_study.py

This produces the forecast panels, the skill-vs-lead-time curve, and theevent-detection scores; figures are saved in the output directory.
  
Run the pilot benchmark:

bash

python run_benchmark.py

python analyze_benchmark.py

The benchmark runner saves one small result file per forecast date under `results/`, allowing interrupted sessions to resume without repeating completed forecasts


## Repository layout

- src/config.py - region, variables, grid constants
- src/credentials.py - safe credential handling (no keys in source)
- src/forecast.py - Aurora forecast pipeline
- src/satellite.py - Sentinel-5P AAI and MODIS AOD retrieval
- src/validation.py - regridding, spatial/temporal correlation, lead-time skill, event detection
- src/plots.py - figure generation (saves to figures/)
- run_case_study.py - end-to-end example

## Results

Multi-season benchmark (headline result). Aurora dust-forecast skill against MODIS
across two Harmattan seasons (26 dates), with 95% bootstrap confidence intervals.
Skill is significant at 1-2 day lead and decays to indistinguishable-from-zero by day 3+:

![Aurora dust forecast skill vs lead time - multi-season benchmark with 95% CIs](benchmark_skill_vs_leadtime.png)

Single-season skill-vs-lead-time curve (January 2024 case study, for reference):

![Aurora dust forecast skill vs lead time over Nigeria (January 2024)](skill_vs_leadtime.png)

Example 5-day dust forecast evolution over Nigeria (15 January 2024 initialization),
showing the Harmattan plume building in the north and being transported over the region:

![Aurora PM10 5-day dust forecast evolution over Nigeria, 15 January 2024](forecast_5day_2024.png)

Validation reference: CAMS dust aerosol optical depth compared with MODIS Combined
aerosol optical depth over Nigeria (temporal composites, January 2024):

![CAMS dust-AOD vs MODIS Combined AOD composite over Nigeria](cams_vs_modis_2024.png)

All figures are regenerated by running `python run_case_study.py`.


## Limitations

This repository presents a pilot benchmark, not an operational warning service. Results are based on two Harmattan seasons
and 26 forecast initialization dates.
While the benchmark shows statistically significant 1–2 day forecast skill, confidence intervals 
remain wide and will be tightened through a larger multi-season evaluation.

Aurora outputs surface particulate matter, while MODIS provides column aerosol optical depth. The comparison is therefore 
interpreted as regional temporal agreement and event-detection skill, not exact concentration-level validation.

Satellite retrieval over bright desert surfaces and polar-orbiter coverage gaps introduce uncertainty in the observational reference, 
especially over northern Nigeria. The benchmark addresses this through the MODIS Combined Dark Target / Deep Blue product, QA filtering, 
and temporal compositing, but the limitation remains important.

The repository does not redistribute Aurora model weights or raw third-party datasets. Users are responsible for complying 
with the licence terms of Microsoft Aurora, Copernicus/CAMS, NASA Earthdata, and Google Earth Engine.


## Authorship and provenance

This repository was developed by Samson Adekoya as part of an MSc research project in Data and Information Science at the University of Ibadan.

To the best of the author's knowledge, this is the first open, reproducible evaluation of the Aurora Earth-system
foundation model for Harmattan dust forecasting over Nigeria.

AI-assisted coding was used during implementation. The research question, scientific design, data-source selection, v
alidation methodology, interpretation, and documentation were led by the author


## Citation and licence

If you use this repository, code, or benchmark, please cite it using the `CITATION.cff` file.

Code in this repository is released under the MIT License.

Benchmark outputs and documentation are intended for release under CC BY 4.0.

This repository does not redistribute Aurora model weights. Users are responsible for complying with the licence terms of the Microsoft Aurora model and all third-party data providers.

## Acknowledgements

Copernicus Atmosphere Monitoring Service (CAMS); NASA (MODIS, Earthdata);
ESA/Copernicus (Sentinel-5P); Microsoft Research (Aurora).
