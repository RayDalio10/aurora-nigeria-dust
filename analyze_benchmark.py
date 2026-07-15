"""Aggregate saved benchmark records and print/plot the skill summary.

CPU-only. Run after run_benchmark.py has collected dates.
Run:  python analyze_benchmark.py
"""
import numpy as np
import matplotlib.pyplot as plt

import benchmark_store as store
import benchmark_metrics as bm

LEADS = bm.LEADS


def main():
    records = store.load_all_results()
    print(f"Loaded {len(records)} date records.")
    if not records:
        print("No results yet. Run run_benchmark.py first.")
        return

    cov = bm.coverage_summary(records)
    print("Coverage:", cov)

    for region in ("whole", "north"):
        print(f"\n=== Skill by lead ({region} domain) ===")
        skill = bm.skill_by_lead(records, region=region)
        for lead in LEADS:
            s = skill[lead]
            print(f"Lead {lead}d (n={s['n']}): "
                  f"Pearson {s['pearson']:.3f} "
                  f"CI[{s['pearson_ci'][0]:.2f},{s['pearson_ci'][1]:.2f}]  "
                  f"Spearman {s['spearman']:.3f} "
                  f"CI[{s['spearman_ci'][0]:.2f},{s['spearman_ci'][1]:.2f}]")

    ev = bm.event_detection(records, lead=2, percentile=70, region="whole")
    print("\nEvent detection (2-day lead, whole domain):", ev)

    # Skill-vs-lead figure with confidence intervals (whole domain)
    skill = bm.skill_by_lead(records, region="whole")
    pear = [skill[l]["pearson"] for l in LEADS]
    plo = [skill[l]["pearson_ci"][0] for l in LEADS]
    phi = [skill[l]["pearson_ci"][1] for l in LEADS]
    yerr = [np.array(pear) - np.array(plo), np.array(phi) - np.array(pear)]

    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(LEADS, pear, yerr=yerr, fmt="o-", capsize=4, label="Pearson (95% CI)")
    ax.set_xlabel("Forecast lead time (days)")
    ax.set_ylabel("Correlation with MODIS")
    ax.set_title("Aurora dust forecast skill vs lead time - multi-season benchmark")
    ax.set_ylim(-0.6, 1)
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("figures/benchmark_skill_vs_leadtime.png", dpi=150, bbox_inches="tight")
    print("\nSaved figures/benchmark_skill_vs_leadtime.png")


if __name__ == "__main__":
    main()
