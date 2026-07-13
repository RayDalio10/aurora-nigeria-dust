
This produces the forecast panels, the skill-vs-lead-time curve, and the
event-detection scores; figures are saved in figures/.

## Repository layout

- src/config.py - region, variables, grid constants
- src/credentials.py - safe credential handling (no keys in source)
- src/forecast.py - Aurora forecast pipeline
- src/satellite.py - Sentinel-5P AAI and MODIS AOD retrieval
- src/validation.py - regridding, spatial/temporal correlation, lead-time skill, event detection
- src/plots.py - figure generation (saves to figures/)
- run_case_study.py - end-to-end example

## Results

![Skill vs lead time](figures/skill_vs_leadtime.png)

## Limitations

Preliminary single-season results (small sample). Aurora outputs surface
particulate matter while satellite references are column measures - an
acknowledged variable mismatch handled via regional/temporal framing. Satellite
retrieval over bright desert surfaces and polar-orbiter coverage gaps add
observational uncertainty; addressed via the Combined product, QA filtering,
and temporal compositing.

## Citation and licence

Code released under the MIT licence (see LICENSE). The Aurora model weights are
licensed CC-BY-NC-SA (non-commercial) by Microsoft; respect that licence.
AI-assisted coding was used during development; scientific design and
interpretation are the author's own.

## Acknowledgements

Copernicus Atmosphere Monitoring Service (CAMS); NASA (MODIS, Earthdata);
ESA/Copernicus (Sentinel-5P); Microsoft Research (Aurora).
