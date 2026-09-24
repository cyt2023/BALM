#!/usr/bin/env python3
"""
run_sensitivity.py — 补充实验：消融、敏感性、子集分析。

子命令:
  bigbudget   更高预算点（与 TTL-60m 的内存水平对齐）
  seeds       多个随机种子（用于时延/SLO 指标的误差棒）
  ablation    BALM-JIT 各组件消融
  lowfreq     仅保留低频应用（平均间隔 >= 5 分钟）的子集，检验方法不是
              只靠"每分钟都在跑"的定时任务取巧
  coldpen     冷启动时延标度敏感性
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy import sparse

import sim
from policies_plan import BALMJIT, BALMPlanner, OracleKnapsack, PlannerLRU, PlannerPopularity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(OUT, "sensitivity.csv")


def append(rows, row):
    rows.append(row)
    pd.DataFrame(rows).to_csv(CSV, index=False)


def trace_lowfreq(ndays=None, seed=7, min_gap=5.0):
    """只保留平均激活间隔 >= min_gap 分钟的应用。"""
    tr = sim.load_trace(ndays=ndays, seed=seed)
    c = tr["counts"].tocsr()
    keep = []
    for a in range(c.shape[0]):
        s, e = c.indptr[a], c.indptr[a + 1]
        if e - s < 2:
            continue
        gaps = np.diff(c.indices[s:e])
        if gaps.mean() >= min_gap:
            keep.append(a)
    keep = np.array(keep)
    tr2 = dict(tr)
    tr2["counts"] = tr["counts"].tocsc()[keep, :].tocsc()
    for k in ("mem_mb", "mean_ms", "q_ms", "trigger", "mean_s", "q_s"):
        tr2[k] = tr[k][keep]
    tr2["n_apps"] = len(keep)
    return tr2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["bigbudget", "seeds", "ablation", "lowfreq", "coldpen"])
    args = ap.parse_args()

    rows = pd.read_csv(CSV).to_dict("records") if os.path.exists(CSV) else []

    if args.part == "bigbudget":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        for b in (1792, 2048):
            B = b * 1024.0
            for name, pol in [("Planner-LRU", PlannerLRU(n, mem, B)),
                              ("Planner-Popularity", PlannerPopularity(n, mem, B)),
                              ("BALM-JIT (ours)", BALMJIT(n, mem, B)),
                              ("Oracle knapsack", OracleKnapsack(n, mem, B))]:
                t0 = time.time()
                r = sim.simulate(trace, pol, seed=0)
                r.update(part="bigbudget", policy=name, budget_gb=b, seed=0,
                         full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"{name:>20s} B={b}GB cold/1k={r['cold_ratio_per_1k']:.4f} "
                      f"mem={r['mean_prewarm_mem_gb']:.0f}GB ({time.time()-t0:.0f}s)", flush=True)

    elif args.part == "seeds":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        for seed in (1, 2):
            for name, pol in [("Scale-to-zero", sim.NoKeepAlive(n, mem)),
                              ("TTL-1m", sim.FixedTTL(n, mem, ttl=1)),
                              ("TTL-5m", sim.FixedTTL(n, mem, ttl=5)),
                              ("TTL-20m", sim.FixedTTL(n, mem, ttl=20)),
                              ("TTL-60m", sim.FixedTTL(n, mem, ttl=60)),
                              ("Historic-TTL", sim.HybridHistogram(n, mem))]:
                r = sim.simulate(trace, pol, seed=seed)
                r.update(part="seeds", policy=name, budget_gb=-1, seed=seed,
                         full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"[seed {seed}] {name:>16s} cold/1k={r['cold_ratio_per_1k']:.3f} "
                      f"p99.9={r['p999_latency_s']:.2f}s slo={r['slo_violation_rate']:.4f}", flush=True)
            for b in (1024, 1536):
                B = b * 1024.0
                for name, pol in [("Planner-LRU", PlannerLRU(n, mem, B)),
                                  ("Planner-Popularity", PlannerPopularity(n, mem, B)),
                                  ("BALM-JIT (ours)", BALMJIT(n, mem, B)),
                                  ("Oracle knapsack", OracleKnapsack(n, mem, B))]:
                    t0 = time.time()
                    r = sim.simulate(trace, pol, seed=seed)
                    r.update(part="seeds", policy=name, budget_gb=b, seed=seed,
                             full_warm_gb=mem.sum() / 1024, n_apps=n)
                    append(rows, r)
                    print(f"[seed {seed}] {name:>20s} B={b}GB cold/1k={r['cold_ratio_per_1k']:.4f} "
                          f"p99.9={r['p999_latency_s']:.2f}s ({time.time()-t0:.0f}s)", flush=True)

    elif args.part == "ablation":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        B = 1024.0 * 1024.0
        variants = [
            ("BALM-JIT (full)", dict()),
            ("- no per-app gap model", dict(per_app=False)),
            ("- no memory awareness", dict(mem_aware=False)),
            ("- coarse gap bins (<=15min)", dict(max_gap=15)),
            ("- no global prior", dict(use_global_prior=False, per_app=True)),
        ]
        for name, kw in variants:
            r = sim.simulate(trace, BALMJIT(n, mem, B, **kw), seed=0)
            r.update(part="ablation", policy=name, budget_gb=1024, seed=0,
                     full_warm_gb=mem.sum() / 1024, n_apps=n)
            append(rows, r)
            print(f"{name:>28s} cold/1k={r['cold_ratio_per_1k']:.4f} "
                  f"mem={r['mean_prewarm_mem_gb']:.0f}GB", flush=True)
        # 另附：季节+动量的规划器（早期设计），作为对比
        r = sim.simulate(trace, BALMPlanner(n, mem, B), seed=0)
        r.update(part="ablation", policy="BALM (season+momentum)", budget_gb=1024, seed=0,
                 full_warm_gb=mem.sum() / 1024, n_apps=n)
        append(rows, r)
        print(f"{'BALM (season+momentum)':>28s} cold/1k={r['cold_ratio_per_1k']:.4f}", flush=True)

    elif args.part == "lowfreq":
        trace = trace_lowfreq(seed=7, min_gap=5.0)
        n, mem = trace["n_apps"], trace["mem_mb"]
        print(f"低频子集: {n} apps, fully-warm={mem.sum()/1024:.0f} GB, "
              f"invocations={trace['counts'].sum():,.0f}")
        for b in (128, 256, 512):
            B = b * 1024.0
            for name, pol in [("Planner-LRU", PlannerLRU(n, mem, B)),
                              ("Planner-Popularity", PlannerPopularity(n, mem, B)),
                              ("BALM-JIT (ours)", BALMJIT(n, mem, B)),
                              ("Oracle knapsack", OracleKnapsack(n, mem, B)),
                              ("TTL-20m", sim.FixedTTL(n, mem, ttl=20))]:
                r = sim.simulate(trace, pol, seed=0)
                r.update(part="lowfreq", policy=name, budget_gb=b, seed=0,
                         full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"B={b:5d}GB {name:>20s} cold/1k={r['cold_ratio_per_1k']:7.3f} "
                      f"mem={r['mean_prewarm_mem_gb']:7.1f}GB", flush=True)

    elif args.part == "coldpen":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        B = 1024.0 * 1024.0
        for med in (0.25, 0.5, 1.0, 2.0, 4.0):
            for name, pol in [("Planner-LRU", PlannerLRU(n, mem, B)),
                              ("BALM-JIT (ours)", BALMJIT(n, mem, B)),
                              ("TTL-20m", sim.FixedTTL(n, mem, ttl=20))]:
                r = sim.simulate(trace, pol, seed=0, cold_median_s=med)
                r.update(part="coldpen", policy=name, budget_gb=1024, seed=0,
                         cold_median_s=med, full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"cold_median={med}s {name:>20s} slo={r['slo_violation_rate']:.4f} "
                      f"p99.9={r['p999_latency_s']:.2f}s cold/1k={r['cold_ratio_per_1k']:.3f}",
                      flush=True)


if __name__ == "__main__":
    main()
