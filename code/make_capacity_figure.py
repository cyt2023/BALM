#!/usr/bin/env python3
"""make_capacity_figure.py — 绘制“单一总内存上限”下的策略对比图。

输入: results/capacity_<tag>.csv
输出: figures/fig_capacity.pdf / .png
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_capacity import PROTOCOL_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")

os.makedirs(FIG, exist_ok=True)

STYLE = {
    "BALM-JIT (ours)": ("D", "#228833", "BALM (ours)"),
    "Planner-Learned": ("P", "#AA3377", "Learned predictor"),
    "Planner-LRU": ("s", "#EE6677", "LRU planner"),
    "Planner-Popularity": ("^", "#CCBB44", "Popularity planner"),
    "Oracle knapsack": ("*", "#66CCEE", "Oracle (non-causal upper bound)"),
    "TTL-20m": ("o", "#444444", "TTL-20m (lease, reference)"),
    "Historic-TTL (ATC'20-style)": ("v", "#999999", "Historic-TTL (reference)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(RES, f"capacity_{args.tag}.csv"))
    if "protocol_version" not in df or not df["protocol_version"].eq(PROTOCOL_VERSION).all():
        raise ValueError("Capacity results predate the causal-eviction fix; rerun with a new --tag")
    if args.tag == "causal-full" and (len(df) != 14 or
                                      set(df.capacity_frac_of_reference) != {0.8, 1.0} or
                                      not df.request_seed.eq(0).all() or
                                      df.groupby(["capacity_frac_of_reference", "policy"]).size().ne(1).any()):
        raise ValueError("Full-trace capacity results are incomplete")
    df = df[df["capacity_violations"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for pol, (mk, col, lab) in STYLE.items():
        d = df[df.policy == pol]
        if not len(d):
            continue
        g = d.groupby("total_capacity_gb")["cold_ratio_per_1k"]
        x, y, lo, hi = [], [], [], []
        for cap, vals in g:
            x.append(cap); y.append(vals.mean()); lo.append(vals.min()); hi.append(vals.max())
        ax.plot(x, y, mk + "-", color=col, ms=4, lw=1.0, label=lab)
        if len(x) > 1:
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("Hard total capacity $C$ (GB)")
    ax.set_ylabel("Cold starts per 1,000 served requests")
    ax.legend(fontsize=5.0, frameon=True, framealpha=0.9, loc="upper right",
              handlelength=1.4, borderpad=0.3, labelspacing=0.25)

    ax2 = axes[1]
    for pol, (mk, col, lab) in STYLE.items():
        d = df[df.policy == pol]
        if not len(d):
            continue
        g = d.groupby("total_capacity_gb")["service_completion_rate"]
        ax2.plot([c for c, _ in g], [v.mean() * 100 for _, v in g], mk + "-",
                 color=col, ms=4, lw=1.0)
    ax2.set_xlabel("Hard total capacity $C$ (GB)")
    ax2.set_ylabel("Service completion rate (%)")
    ax2.set_ylim(90, 100.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_capacity.{ext}"))
    plt.close(fig)
    print("-> figures/fig_capacity.pdf / .png")


if __name__ == "__main__":
    main()
