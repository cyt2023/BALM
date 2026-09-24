#!/usr/bin/env python3
"""
run_hazard_main.py — 修复 Eq.(7) 后的主实验（hazard 版本），并同时跑 PMF 消融。

背景：初版代码用边缘 PMF P(G=lag) 当作"本分钟活跃概率"，正确量应为条件风险
P(G=lag | G>=lag)。本脚本重跑全部主结果，并把 PMF 版本作为对照一并记录。

用法: python3 run_hazard_main.py [--budgets 64,128,...] [--seeds 0]
输出: results/main_hazard.csv (可重复运行，已完成的配置会跳过)
"""
import argparse
import os
import time
import numpy as np
import pandas as pd

import sim
from policies_plan import BALMJIT, OracleKnapsack, PlannerLRU, PlannerPopularity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
CSV = os.path.join(OUT, "main_hazard.csv")

BUDGETS_GB = [64, 128, 256, 512, 768, 1024, 1536, 1792]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=str, default=None)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--ndays", type=int, default=None)
    ap.add_argument("--napps", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--skip-lease", action="store_true")
    ap.add_argument("--pmf-budgets", type=str, default="256,1024,1536")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else BUDGETS_GB
    seeds = [int(s) for s in args.seeds.split(",")]
    pmf_budgets = {int(b) for b in args.pmf_budgets.split(",")}
    global CSV
    if args.out:
        CSV = os.path.join(OUT, args.out)

    trace = sim.load_trace(ndays=args.ndays, nsample_apps=args.napps, seed=7)
    n, mem = trace["n_apps"], trace["mem_mb"]
    full_warm = mem.sum() / 1024.0
    print(f"trace: {n} apps, {trace['n_minutes']} minutes, "
          f"fully-warm={full_warm:.0f} GB", flush=True)

    rows = []
    done = set()
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        rows = prev.to_dict("records")
        done = {(r["policy"], r["budget_gb"], r["seed"]) for r in rows}
        print(f"已有 {len(rows)} 条，跳过重复配置", flush=True)

    def record(res, policy, budget_gb, seed, extra=None):
        row = dict(res)
        row.update(policy=policy, budget_gb=budget_gb, seed=seed,
                   full_warm_gb=full_warm, n_apps=n)
        if extra:
            row.update(extra)
        rows.append(row)
        pd.DataFrame(rows).to_csv(CSV, index=False)

    t0 = time.time()
    for seed in seeds:
        lease = [] if args.skip_lease else [
            ("Scale-to-zero", sim.NoKeepAlive(n, mem)),
            ("TTL-1m", sim.FixedTTL(n, mem, ttl=1)),
            ("TTL-5m", sim.FixedTTL(n, mem, ttl=5)),
            ("TTL-20m", sim.FixedTTL(n, mem, ttl=20)),
            ("TTL-60m", sim.FixedTTL(n, mem, ttl=60)),
            ("Historic-TTL", sim.HybridHistogram(n, mem)),
        ]
        for name, pol in lease:
            key = (name, -1, seed)
            if key in done:
                continue
            t1 = time.time()
            res = sim.simulate(trace, pol, seed=seed)
            record(res, name, -1, seed)
            print(f"[s{seed}] {name:>16s} cold/1k={res['cold_ratio_per_1k']:7.3f} "
                  f"mem={res['mean_prewarm_mem_gb']:7.1f} "
                  f"prewarm/1k={res['prewarm_per_1k']:8.3f} inits/1k={res['total_inits_per_1k']:8.3f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

        for b in budgets:
            B = b * 1024.0
            planners = [
                ("Planner-LRU", PlannerLRU(n, mem, B)),
                ("Planner-Popularity", PlannerPopularity(n, mem, B)),
                ("BALM-JIT (hazard)", BALMJIT(n, mem, B)),
                ("Oracle knapsack", OracleKnapsack(n, mem, B)),
            ]
            if b in pmf_budgets:
                planners.append(("BALM-JIT (PMF)", BALMJIT(n, mem, B, use_hazard=False)))
            for name, pol in planners:
                key = (name, b, seed)
                if key in done:
                    continue
                t1 = time.time()
                res = sim.simulate(trace, pol, seed=seed)
                record(res, name, b, seed)
                print(f"[s{seed}] {name:>20s} B={b:5d} cold/1k={res['cold_ratio_per_1k']:7.3f} "
                      f"mem={res['mean_prewarm_mem_gb']:7.1f} util={res['budget_utilization']:.3f} "
                      f"prewarm/1k={res['prewarm_per_1k']:8.3f} inits/1k={res['total_inits_per_1k']:8.3f} "
                      f"k={res['cand_k_mean']:6.0f} gap={res['knapsack_gap_mean']:.4f} "
                      f"({time.time()-t1:.0f}s)", flush=True)
    print(f"\n完成: {CSV}  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
