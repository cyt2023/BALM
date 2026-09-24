#!/usr/bin/env python3
"""04_make_figures.py — Generate plots from experiment outputs (PDF and PNG)。"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": ":", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> figures/{name}.pdf")


def load():
    main = pd.read_csv(os.path.join(RES, "main_hazard.csv"))
    sens = pd.read_csv(os.path.join(RES, "sensitivity.csv")) if \
        os.path.exists(os.path.join(RES, "sensitivity.csv")) else pd.DataFrame()
    char = np.load(os.path.join(RES, "char_data.npz"))
    with open(os.path.join(RES, "characterization.json")) as f:
        cstats = json.load(f)
    with open(os.path.join(RES, "predictability.json")) as f:
        pstats = json.load(f)
    return main, sens, char, cstats, pstats


def fig_characterization(char, cstats, pstats):
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.6))
    mem = char["mem_mb"]
    ax[0, 0].hist(mem, bins=np.logspace(np.log10(30), np.log10(1500), 40))
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_xlabel("Application memory allocation (MB)")
    ax[0, 0].set_ylabel("Applications")
    ax[0, 0].set_title("(a) Memory footprint", fontsize=8)

    gh = char["gap_hist"][1:121]
    ax[0, 1].bar(np.arange(1, 121), gh, width=0.9)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("Inter-arrival gap (minutes)")
    ax[0, 1].set_ylabel("Occurrences")
    ax[0, 1].set_title("(b) Gap distribution (periodicity)", fontsize=8)

    apm = char["active_per_min"]
    ax[1, 0].plot(np.arange(len(apm)) / 1440.0, apm, lw=0.4)
    ax[1, 0].set_xlabel("Trace day")
    ax[1, 0].set_ylabel("Active applications / minute")
    ax[1, 0].set_title("(c) Concurrent activity", fontsize=8)

    strata = list(pstats["strata"].items())
    names = [s[0] for s in strata]
    rates = [s[1]["hit_rate_1min"] * 100 for s in strata]
    ax[1, 1].bar(range(len(names)), rates, color="#4477AA")
    ax[1, 1].set_xticks(range(len(names)))
    ax[1, 1].set_xticklabels(names, rotation=15, fontsize=7)
    ax[1, 1].set_ylabel("Next-activation hit rate (±1 min, %)")
    ax[1, 1].set_ylim(0, 105)
    ax[1, 1].set_title("(d) Predictability by typical gap", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_characterization")


def fig_tradeoff(main):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    lease = main[(main["budget_gb"] == -1) & (main["seed"] == 0)]
    plan = main[(main["budget_gb"] > 0) & (main["seed"] == 0)]

    ax.plot(lease["mean_prewarm_mem_gb"], lease["cold_ratio_per_1k"], "o-",
            color="#444444", ms=3, lw=1.0, label="Lease-based keep-alive")
    for _, r in lease.iterrows():
        ax.annotate(r["policy"].replace("Scale-to-zero", "S2Z"),
                    (r["mean_prewarm_mem_gb"], r["cold_ratio_per_1k"]),
                    textcoords="offset points", xytext=(3, 3), fontsize=5.5)
    styles = {"Planner-LRU": ("s", "#EE6677", "LRU planner"),
              "Planner-Popularity": ("^", "#CCBB44", "Popularity planner"),
              "BALM-JIT (hazard)": ("D", "#228833", "BALM (ours)"),
              "Oracle knapsack": ("*", "#66CCEE", "Oracle (upper bound)")}
    for pol, (mk, col, lab) in styles.items():
        d = plan[plan["policy"] == pol].sort_values("mean_prewarm_mem_gb")
        if len(d) == 0:
            continue
        ax.plot(d["mean_prewarm_mem_gb"], d["cold_ratio_per_1k"], mk + "-",
                color=col, ms=3.5, lw=1.0, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Warm-set memory (GB)")
    ax.set_ylabel("Cold starts per 1,000 invocations")
    ax.legend(fontsize=5.5, loc="upper right", frameon=False)
    save(fig, "fig_tradeoff")


def fig_bars(main):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, b in enumerate((768, 1024, 1536)):
        d = main[(main["budget_gb"] == b) & (main["seed"] == 0)]
        order = ["Planner-LRU", "Planner-Popularity", "BALM-JIT (hazard)", "Oracle knapsack"]
        vals = [float(d[d["policy"] == p]["cold_ratio_per_1k"].iloc[0]) for p in order]
        x = np.arange(len(order)) + i * 0.25
        ax.bar(x, vals, width=0.22, label=f"{b} GB",
               color=["#EE6677", "#CCBB44", "#228833", "#66CCEE"],
               alpha=[0.55, 0.7, 0.95, 0.4][i] if False else 0.85)
    ax.set_xticks(np.arange(4) + 0.25)
    ax.set_xticklabels(["LRU", "Popularity", "BALM-JIT", "Oracle"], fontsize=7)
    ax.set_ylabel("Cold starts per 1,000 invocations")
    ax.set_yscale("log")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "fig_bars")


def fig_ablation(sens):
    d = sens[sens["part"] == "ablation"]
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    d = d.sort_values("cold_ratio_per_1k")
    ax.barh(range(len(d)), d["cold_ratio_per_1k"], color="#228833", alpha=0.85)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([p.replace("BALM-JIT ", "") for p in d["policy"]], fontsize=6)
    ax.set_xlabel("Cold starts per 1,000 invocations (1 TB budget)")
    ax.invert_yaxis()
    save(fig, "fig_ablation")


def fig_lowfreq(sens):
    d = sens[sens["part"] == "lowfreq"]
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for pol, mk, col in [("Planner-LRU", "s", "#EE6677"),
                         ("Planner-Popularity", "^", "#CCBB44"),
                         ("BALM-JIT (ours)", "D", "#228833"),
                         ("Oracle knapsack", "*", "#66CCEE"),
                         ("TTL-20m", "o", "#444444")]:
        dd = d[d["policy"] == pol].sort_values("mean_prewarm_mem_gb")
        if len(dd) == 0:
            continue
        ax.plot(dd["mean_prewarm_mem_gb"], dd["cold_ratio_per_1k"], mk + "-",
                color=col, ms=3.5, lw=1.0, label=pol)
    ax.set_xlabel("Warm-set memory (GB)")
    ax.set_ylabel("Cold starts per 1,000 invocations")
    ax.legend(fontsize=5.5, frameon=False)
    save(fig, "fig_lowfreq")


def fig_coldpen(sens):
    d = sens[sens["part"] == "coldpen"]
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for pol, mk, col in [("TTL-20m", "o", "#444444"), ("Planner-LRU", "s", "#EE6677"),
                         ("BALM-JIT (ours)", "D", "#228833")]:
        dd = d[d["policy"] == pol].sort_values("cold_median_s")
        ax.plot(dd["cold_median_s"], dd["slo_violation_rate"] * 100, mk + "-",
                color=col, ms=3.5, label=pol)
    ax.set_xscale("log")
    ax.set_xlabel("Median cold-start latency (s)")
    ax.set_ylabel("SLO violations (% of invocations)")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "fig_coldpen")


if __name__ == "__main__":
    main_df, sens_df, char, cstats, pstats = load()
    fig_characterization(char, cstats, pstats)
    fig_tradeoff(main_df)
    fig_bars(main_df)
    fig_ablation(sens_df)
    fig_lowfreq(sens_df)
    fig_coldpen(sens_df)
