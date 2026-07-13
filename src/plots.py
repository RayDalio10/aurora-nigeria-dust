"""Figure generation. Every function saves the figure to figures/ AND returns
the matplotlib figure, so results are never lost to a display-only cell.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "figures"


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_dust_map(lons, lats, field, title, fname, cbar_label="PM10 (model units)"):
    """Plot a single gridded dust field (e.g. one Aurora forecast step)."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(lons, lats, field, cmap="YlOrBr", shading="auto")
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_forecast_panels(preds, lons, lats, init_date, fname,
                         steps=(1, 3, 5, 7, 9)):
    """Plot the 5-day forecast as day-by-day panels (each step is 12 h)."""
    _ensure_dir()
    fig, axes = plt.subplots(1, len(steps), figsize=(22, 5))
    for ax, step in zip(axes, steps):
        pm10 = preds[step].surf_vars["pm10"][0, 0].numpy()
        im = ax.pcolormesh(lons, lats, pm10, cmap="YlOrBr", shading="auto")
        ax.set_title(f"Day {(step + 1) // 2}")
        ax.set_xlabel("Lon E")
        ax.set_ylabel("Lat N")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Aurora PM10 dust forecast over Nigeria - 5-day evolution "
                 f"(from {init_date})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_two_maps(grid_a, grid_b, title_a, title_b, fname, extent=(2, 15, 3, 16)):
    """Side-by-side comparison of two regridded maps (e.g. forecast vs obs)."""
    _ensure_dir()
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im0 = ax[0].imshow(grid_a, origin="lower", cmap="YlOrBr", extent=extent, aspect="auto")
    ax[0].set_title(title_a)
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(grid_b, origin="lower", cmap="YlOrBr", extent=extent, aspect="auto")
    ax[1].set_title(title_b)
    fig.colorbar(im1, ax=ax[1])
    for a in ax:
        a.set_xlabel("Lon E")
        a.set_ylabel("Lat N")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_skill_vs_leadtime(leads, pearson, spearman, fname="skill_vs_leadtime.png",
                           title="Aurora dust forecast skill vs lead time - Nigeria, Jan 2024"):
    """Plot the forecast skill-decay curve (the headline result) and save it."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(leads, pearson, "o-", label="Pearson")
    ax.plot(leads, spearman, "s--", label="Spearman")
    ax.set_xlabel("Forecast lead time (days)")
    ax.set_ylabel("Correlation with MODIS")
    ax.set_title(title)
    ax.set_ylim(-0.4, 1)
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_scatter(x, y, xlabel, ylabel, fname, title="", log=False):
    """Scatter of two paired series (raw or log), saved to file."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    xv, yv = (np.log(x), np.log(y)) if log else (x, y)
    ax.scatter(xv, yv, s=12, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_three_year_day3(runs, fname, day_step=5):
    """Three-panel Day-3 comparison across years.

    runs: list of (label, preds, atm) tuples. day_step is the rollout step
    for ~Day 3 (step 5). Saves a side-by-side comparison figure.
    """
    _ensure_dir()
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6))
    for ax, (label, preds, atm) in zip(axes, runs):
        pm10 = preds[day_step].surf_vars["pm10"][0, 0].numpy()
        im = ax.pcolormesh(atm.longitude.values, atm.latitude.values, pm10,
                           cmap="YlOrBr", shading="auto")
        ax.set_title(f"{label}  (Day 3)")
        ax.set_xlabel("Lon E")
        ax.set_ylabel("Lat N")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Aurora PM10 dust over Nigeria - Day 3 forecast, three Harmattan seasons",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    return fig


def plot_scatter_map(lons, lats, values, title, fname,
                     cbar_label="value", extent=(2, 15, 3, 16)):
    """Scatter plot of per-pixel values on a lon/lat map (e.g. satellite AAI)."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(lons, lats, c=values, cmap="YlOrBr", s=40, marker="s")
    ax.set_title(title)
    ax.set_xlabel("Longitude E")
    ax.set_ylabel("Latitude N")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
    return fig
    
