#!/usr/bin/env python3
"""
run_main.py — 主实验：在完整 Azure Functions 2019 trace 上比较保活/规划策略。

用法:
  python3 run_main.py                 # 全部配置
  python3 run_main.py --quick         # 快速子集（调试用）
结果追加写入 results/main_results.csv（可重复运行，已完成的配置会跳过）。
"""
import argparse
import os
import time
import numpy as np
import pandas as pd

import sim
from policies_plan import (BALMJIT, BALMPlanner, OracleKnapsack, PlannerLRU,
                           PlannerPopularity)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(OUT, "main_results.csv")

BUDGETS_GB = [64, 128, 256, 512]
SEEDS = [0, 1, 2]


def build_policies(n_apps, mem, budget_mb, seed):
    return [
        ("Scale-to-zero", sim.NoKeepAlive(n_apps, mem)),
        ("TTL-1m", sim.FixedTTL(n_apps, mem, ttl=1)),
        ("TTL-5m", sim.FixedTTL(n_apps, mem, ttl=5)),
        ("TTL-20m", sim.FixedTTL(n_apps, mem, ttl=20)),
        ("TTL-60m", sim.FixedTTL(n_apps, mem, ttl=60)),
        ("Historic-TTL", sim.HybridHistogram(n_apps, mem)),
    ], [
        ("Planner-LRU", PlannerLRU(n_apps, mem, budget_mb)),
        ("Planner-Popularity", PlannerPopularity(n_apps, mem, budget_mb)),
        ("BALM-JIT (ours)", BALMJIT(n_apps, mem, budget_mb)),
        ("Oracle knapsack", OracleKnapsack(n_apps, mem, budget_mb)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ndays", type=int, default=None)
    ap.add_argument("--napps", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--budgets", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    global SEEDS, BUDGETS_GB, CSV
    if args.seeds:
        SEEDS = [int(s) for s in args.seeds.split(",")]
    if args.budgets:
        BUDGETS_GB = [int(b) for b in args.budgets.split(",")]
    if args.out:
        CSV = os.path.join(OUT, args.out)

    ndays = 3 if args.quick else args.ndays
    napps = 4000 if args.quick else args.napps
    trace = sim.load_trace(ndays=ndays, nsample_apps=napps, seed=7)
    n, mem = trace["n_apps"], trace["mem_mb"]
    full_warm_gb = mem.sum() / 1024.0
    print(f"trace: {n} apps, {trace['n_minutes']} minutes, "
          f"{trace['counts'].sum():,.0f} invocations, fully-warm={full_warm_gb:.0f} GB")

    done = set()
    rows = []
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        rows = prev.to_dict("records")
        done = {tuple(r[k] for k in ("policy", "budget_gb", "seed")) for r in rows}
        print(f"已有 {len(rows)} 条结果，跳过重复配置")

    def record(res, budget_gb, seed, extra=None):
        row = dict(res)
        row["budget_gb"] = budget_gb
        row["seed"] = seed
        row["full_warm_gb"] = full_warm_gb
        row["n_apps"] = n
        if extra:
            row.update(extra)
        rows.append(row)
        pd.DataFrame(rows).to_csv(CSV, index=False)

    t_start = time.time()
    for seed in SEEDS:
        # ---- 无预算约束的经典保活策略 ----
        lease_pols, planner_pols = build_policies(n, mem, 0, seed)
        for name, pol in lease_pols:
            if (name, -1, seed) in done:
                continue
            t0 = time.time()
            res = sim.simulate(trace, pol, seed=seed)
            res["policy"] = name
            record(res, -1, seed)
            print(f"[seed {seed}] {name:>20s} cold/1k={res['cold_ratio_per_1k']:7.2f} "
                  f"mem={res['mean_prewarm_mem_gb']:7.1f}GB  ({time.time()-t0:.0f}s)", flush=True)

        # ---- 预算约束下的逐分钟规划策略 ----
        for budget_gb in BUDGETS_GB:
            budget_mb = budget_gb * 1024.0
            _, planner_pols = build_policies(n, mem, budget_mb, seed)
            for name, pol in planner_pols:
                if (name, budget_gb, seed) in done:
                    continue
                t0 = time.time()
                res = sim.simulate(trace, pol, seed=seed)
                res["policy"] = name
                record(res, budget_gb, seed)
                print(f"[seed {seed}] {name:>20s} B={budget_gb:5d}GB "
                      f"cold/1k={res['cold_ratio_per_1k']:7.2f} "
                      f"mem={res['mean_prewarm_mem_gb']:7.1f}GB  ({time.time()-t0:.0f}s)",
                      flush=True)

    print(f"\n完成，用时 {(time.time()-t_start)/60:.1f} 分钟 -> {CSV}")


if __name__ == "__main__":
    main()
