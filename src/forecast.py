"""Aurora dust-forecast pipeline for Nigeria.

Downloads CAMS input, crops to the study domain, builds the Aurora input
batch, and runs inference to produce a multi-day dust forecast.
"""
import os
import zipfile
import pickle

import numpy as np
import torch
import xarray as xr
import cdsapi
from huggingface_hub import hf_hub_download
from aurora import AuroraAirPollution, Batch, Metadata, rollout

from config import NIGERIA_BOUNDS, CAMS_VARIABLES, PRESSURE_LEVELS

STATIC_FILE = "aurora-0.4-air-pollution-static.pickle"
CHECKPOINT = "aurora-0.4-air-pollution.ckpt"


def download_cams(date, out_zip):
    """Download one CAMS snapshot for the given date (YYYY-MM-DD)."""
    if os.path.exists(out_zip):
        print(f"{out_zip} already exists, skipping download.")
        return
    client = cdsapi.Client()
    client.retrieve(
        "cams-global-atmospheric-composition-forecasts",
        {
            "variable": CAMS_VARIABLES,
            "pressure_level": PRESSURE_LEVELS,
            "date": f"{date}/{date}",
            "time": ["00:00", "12:00"],
            "leadtime_hour": "0",
            "type": "forecast",
            "data_format": "netcdf_zip",
        },
        out_zip,
    )
    print(f"Downloaded {out_zip}")


def load_and_crop(zip_path, extract_dir):
    """Unzip CAMS data and crop to the Nigeria domain.

    Returns (surface_dataset, atmospheric_dataset, path_to_surface_nc).
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    b = NIGERIA_BOUNDS
    surf = xr.open_dataset(f"{extract_dir}/data_sfc.nc").isel(forecast_period=0)
    atm = xr.open_dataset(f"{extract_dir}/data_plev.nc").isel(forecast_period=0)
    surf = surf.sel(latitude=slice(b["lat_max"], b["lat_min"]),
                    longitude=slice(b["lon_min"], b["lon_max"]))
    atm = atm.sel(latitude=slice(b["lat_max"], b["lat_min"]),
                  longitude=slice(b["lon_min"], b["lon_max"]))
    return surf, atm, f"{extract_dir}/data_sfc.nc"


def crop_static(static_path, full_sfc_path):
    """Crop the global Aurora static file to the Nigeria domain."""
    with open(static_path, "rb") as f:
        static_full = pickle.load(f)
    ds = xr.open_dataset(full_sfc_path)
    b = NIGERIA_BOUNDS
    lat_idx = np.where((ds.latitude.values <= b["lat_max"]) &
                       (ds.latitude.values >= b["lat_min"]))[0]
    lon_idx = np.where((ds.longitude.values >= b["lon_min"]) &
                       (ds.longitude.values <= b["lon_max"]))[0]
    return {
        k: np.asarray(v)[..., lat_idx.min():lat_idx.max() + 1,
                         lon_idx.min():lon_idx.max() + 1]
        for k, v in static_full.items()
    }


def build_batch(surf, atm, static_ng):
    """Assemble the Aurora input Batch from cropped datasets."""
    def s(name):
        return torch.from_numpy(surf[name].values[None])

    def a(name):
        return torch.from_numpy(atm[name].values[None])

    return Batch(
        surf_vars={
            "2t": s("t2m"), "10u": s("u10"), "10v": s("v10"), "msl": s("msl"),
            "pm1": s("pm1"), "pm2p5": s("pm2p5"), "pm10": s("pm10"),
            "tcco": s("tcco"), "tc_no": s("tc_no"), "tcno2": s("tcno2"),
            "gtco3": s("gtco3"), "tcso2": s("tcso2"),
        },
        static_vars={k: torch.from_numpy(v.copy()) for k, v in static_ng.items()},
        atmos_vars={
            "t": a("t"), "u": a("u"), "v": a("v"), "q": a("q"), "z": a("z"),
            "co": a("co"), "no": a("no"), "no2": a("no2"),
            "go3": a("go3"), "so2": a("so2"),
        },
        metadata=Metadata(
            lat=torch.from_numpy(atm.latitude.values),
            lon=torch.from_numpy(atm.longitude.values),
            time=(surf.forecast_reference_time.values
                  .astype("datetime64[s]").tolist()[-1],),
            atmos_levels=tuple(int(l) for l in atm.pressure_level.values),
        ),
    )


_MODEL = None


def get_model(device="cuda"):
    """Load the pre-trained Aurora air-pollution model once and cache it."""
    global _MODEL
    if _MODEL is None:
        _MODEL = AuroraAirPollution()
        _MODEL.load_checkpoint("microsoft/aurora", CHECKPOINT)
        _MODEL.eval()
        _MODEL = _MODEL.to(device)
    return _MODEL


def run_aurora_forecast(date, steps=10):
    """Run the full forecast pipeline for one date.

    Returns (list_of_forecast_steps, atmospheric_dataset). Each step is 12 h,
    so steps=10 yields a 5-day forecast. The atmospheric dataset carries the
    latitude/longitude coordinates used for plotting and regridding.
    """
    tag = date.replace("-", "")
    download_cams(date, f"cams_{tag}.zip")
    surf, atm, full_sfc = load_and_crop(f"cams_{tag}.zip", f"cams_{tag}")
    static_ng = crop_static(
        hf_hub_download("microsoft/aurora", STATIC_FILE), full_sfc)
    batch = build_batch(surf, atm, static_ng)
    model = get_model()
    with torch.inference_mode():
        preds = [p.to("cpu") for p in rollout(model, batch, steps=steps)]
    print(f"Forecast for {date} done: {len(preds)} steps.")
    return preds, atm
