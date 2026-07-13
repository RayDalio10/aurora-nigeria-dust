"""Safe credential handling. NEVER hard-code API keys in source.

Set the Copernicus ADS credentials via environment variables:
    export CDSAPI_URL="https://ads.atmosphere.copernicus.eu/api"
    export CDSAPI_KEY="your-regenerated-key"
or place them in ~/.cdsapirc (which must be git-ignored).
"""
import os


def write_cdsapirc_from_env():
    """Write ~/.cdsapirc from environment variables if it does not exist."""
    url = os.environ.get("CDSAPI_URL")
    key = os.environ.get("CDSAPI_KEY")
    path = os.path.expanduser("~/.cdsapirc")
    if os.path.exists(path):
        return
    if not (url and key):
        raise RuntimeError(
            "Set CDSAPI_URL and CDSAPI_KEY environment variables, "
            "or create ~/.cdsapirc manually."
        )
    with open(path, "w") as f:
        f.write(f"url: {url}\nkey: {key}\n")
