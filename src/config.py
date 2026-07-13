"""Shared configuration: region bounds, CAMS variables, pressure levels, grid."""
import numpy as np

# Nigeria / West Africa study domain
NIGERIA_BOUNDS = {"lat_max": 16, "lat_min": 3, "lon_min": 2, "lon_max": 15}

# CAMS variables required by the Aurora air-pollution model
CAMS_VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "mean_sea_level_pressure",
    "particulate_matter_1um", "particulate_matter_2.5um",
    "particulate_matter_10um", "total_column_carbon_monoxide",
    "total_column_nitrogen_monoxide", "total_column_nitrogen_dioxide",
    "total_column_ozone", "total_column_sulphur_dioxide",
    "u_component_of_wind", "v_component_of_wind", "temperature",
    "geopotential", "specific_humidity",
    "carbon_monoxide", "nitrogen_dioxide", "nitrogen_monoxide",
    "ozone", "sulphur_dioxide",
]

PRESSURE_LEVELS = ["50", "100", "150", "200", "250", "300",
                   "400", "500", "600", "700", "850", "925", "1000"]

# Earth Engine project id (set your own)
EE_PROJECT = "hidden-ensign-502101-b6"

# Shared validation grid over Nigeria (0.5-degree boxes)
GRID_LON = np.arange(2, 15, 0.5)
GRID_LAT = np.arange(3, 16, 0.5)
