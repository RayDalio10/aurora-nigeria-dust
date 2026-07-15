"""Persistence for the validation benchmark.

Each initialization date's distilled results (region-mean forecast per lead,
MODIS region-mean, coverage flags) are saved as one small JSON file under
results/. The runner uses list_completed_dates() to skip work already done,
which makes the benchmark resumable across free-GPU sessions.

Only distilled numbers are stored here - never the multi-GB raw downloads.
"""
import os
import json
import glob

RESULTS_DIR = "results"


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def result_path(date):
    """Path for one date's result file, e.g. results/skill_20240115.json."""
    tag = date.replace("-", "")
    return os.path.join(RESULTS_DIR, f"skill_{tag}.json")


def is_done(date):
    """True if this date already has a saved result."""
    return os.path.exists(result_path(date))


def save_result(date, record):
    """Save one date's result record (a dict) as JSON."""
    _ensure_dir()
    record = dict(record)
    record["date"] = date
    with open(result_path(date), "w") as f:
        json.dump(record, f, indent=2)


def load_result(date):
    """Load one date's result record, or None if not present."""
    path = result_path(date)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_completed_dates():
    """Return the sorted list of dates that already have saved results."""
    _ensure_dir()
    dates = []
    for p in glob.glob(os.path.join(RESULTS_DIR, "skill_*.json")):
        try:
            with open(p) as f:
                dates.append(json.load(f)["date"])
        except Exception:
            continue
    return sorted(dates)


def load_all_results():
    """Return all saved result records as a list of dicts, sorted by date."""
    recs = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "skill_*.json"))):
        try:
            with open(p) as f:
                recs.append(json.load(f))
        except Exception:
            continue
    return recs
