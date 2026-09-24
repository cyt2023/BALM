#!/usr/bin/env python3
"""run_ablation.py — 消融实验（1 TB 预算），含 mean/worst LP gap 与预测器审计。

输出: results/revision_ablation.csv
"""
import os
import time

import numpy as np
import pandas as pd

import sim
from policies_plan import BALMJIT, BALMPlanner

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
CSV = os.path.join(OUT, "revision_ablation.csv")

VARIANTS = [
    ("BALM with hazard estimator (Eq. 7)", dict()),
    ("with marginal gap PMF instead", dict(use_hazard=False)),
    ("without per-application histogram", dict(per_app=False)),
    ("without memory awareness", dict(mem_aware=False)),
    ("with 15-min gap resolution", dict(max_gap=15)),
]


def main():
    trace = sim.load_trace(seed=7)
    n, mem = trace["n_apps"], trace["mem_mb"]
    B = 1024 * 1024.0
    rows = []
    for label, kw in VARIANTS:
        t0 = time.time()
        r = sim.simulate(trace, BALMJIT(n, mem, B, **kw), seed=0)
        r.update(policy=label, budget_gb=1024, seed=0, n_apps=n,
                 full_warm_gb=mem.sum() / 1024)
        rows.append(r)
        print(f"{label:34s} cold/1k={r['cold_ratio_per_1k']:7.4f} "
              f"gap_mean={r['knapsack_gap_mean']:.3e} gap_max={r['knapsack_gap_max']:.3e} "
              f"brier={r['predictor_brier']:.4f} auroc={r['predictor_auroc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    # seasonal + momentum 规划器（早期设计）
    t0 = time.time()
    r = sim.simulate(trace, BALMPlanner(n, mem, B), seed=0)
    r.update(policy="seasonal + momentum predictor", budget_gb=1024, seed=0,
             n_apps=n, full_warm_gb=mem.sum() / 1024)
    rows.append(r)
    print(f"{'seasonal + momentum predictor':34s} cold/1k={r['cold_ratio_per_1k']:7.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    # 最小预算点：用于 worst-case LP gap 的表述
    t0 = time.time()
    r64 = sim.simulate(trace, BALMJIT(n, mem, 64 * 1024.0), seed=0)
    r64.update(policy="BALM with hazard estimator (Eq. 7)", budget_gb=64, seed=0,
               n_apps=n, full_warm_gb=mem.sum() / 1024)
    rows.append(r64)
    print(f"{'BALM at 64 GB':34s} cold/1k={r64['cold_ratio_per_1k']:7.4f} "
          f"gap_max={r64['knapsack_gap_max']:.3e} ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(CSV, index=False)
    print(f"\n-> {CSV}")


if __name__ == "__main__":
    main()
