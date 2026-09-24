#!/usr/bin/env python3
"""
run_revision.py — 针对审稿意见补充的实验。

子命令:
  gap       候选窗口 G 的敏感性 (60/120/360/1440 分钟)
  penmode   cold-start 时延模型的敏感性 (memory / constant / independent)
  seeds     随机性指标多 seed 重复 (latency/SLO)
  lowfreq   低频应用子集 + 每应用公平性分布
  unpred    因候选窗口截断而"结构上不可预测"的激活占比
  subminute 亚分钟到达 + 初始化时长的敏感性
"""
import argparse
import os
import time
import numpy as np
import pandas as pd

import sim
from policies_plan import BALMJIT, OracleKnapsack, PlannerLRU, PlannerPopularity
from run_sensitivity import trace_lowfreq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
CSV = os.path.join(OUT, "revision.csv")   # 由 main() 按子命令改写为 revision_<part>.csv


def append(rows, row):
    rows.append(row)
    pd.DataFrame(rows).to_csv(CSV, index=False)


def load_rows():
    return pd.read_csv(CSV).to_dict("records") if os.path.exists(CSV) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["gap", "penmode", "seeds", "lowfreq", "unpred", "subminute"])
    args = ap.parse_args()
    global CSV
    CSV = os.path.join(OUT, f"revision_{args.part}.csv")   # 每个子命令独立文件，便于并行
    rows = load_rows()

    if args.part == "gap":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        for B_GB in (768, 1024):
            B = B_GB * 1024.0
            for G in (60, 120, 360, 1440):
                r = sim.simulate(trace, BALMJIT(n, mem, B, max_gap=G), seed=0)
                r.update(part="gap", policy=f"BALM G={G}", budget_gb=B_GB, seed=0,
                         max_gap=G, full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"G={G:5d} B={B_GB}GB cold/1k={r['cold_ratio_per_1k']:7.4f} "
                      f"k={r['cand_k_mean']:6.0f} pre/1k={r['prewarm_per_1k']:7.3f} "
                      f"inits/1k={r['total_inits_per_1k']:7.3f}", flush=True)

    elif args.part == "penmode":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        B = 1024.0 * 1024.0
        for mode in ("memory", "constant", "independent"):
            for name, pol in [("BALM-JIT (hazard)", BALMJIT(n, mem, B)),
                              ("Planner-LRU", PlannerLRU(n, mem, B)),
                              ("TTL-20m", sim.FixedTTL(n, mem, ttl=20))]:
                r = sim.simulate(trace, pol, seed=0, cold_penalty_mode=mode)
                r.update(part="penmode", policy=name, budget_gb=1024, seed=0,
                         penalty_mode=mode, full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"{mode:>12s} {name:>20s} cold/1k={r['cold_ratio_per_1k']:7.4f} "
                      f"slo/1k={r['cold_slo_per_1k']:7.4f}", flush=True)

    elif args.part == "seeds":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        for seed in (0, 1, 2, 3, 4):
            for B_GB in (1024, 1536):
                B = B_GB * 1024.0
                for name, pol in [("BALM-JIT (hazard)", BALMJIT(n, mem, B)),
                                  ("Planner-LRU", PlannerLRU(n, mem, B)),
                                  ("Planner-Popularity", PlannerPopularity(n, mem, B))]:
                    r = sim.simulate(trace, pol, seed=seed)
                    r.update(part="seeds", policy=name, budget_gb=B_GB, seed=seed,
                             full_warm_gb=mem.sum() / 1024, n_apps=n)
                    append(rows, r)
                    print(f"[seed {seed}] {name:>20s} B={B_GB} slo/1k={r['cold_slo_per_1k']:7.4f} "
                          f"p99.9={r['p999_latency_s']:6.2f}s added_ms={r['mean_cold_added_ms']:.5f}",
                          flush=True)
            r = sim.simulate(trace, sim.FixedTTL(n, mem, ttl=20), seed=seed)
            r.update(part="seeds", policy="TTL-20m", budget_gb=-1, seed=seed,
                     full_warm_gb=mem.sum() / 1024, n_apps=n)
            append(rows, r)
            print(f"[seed {seed}] {'TTL-20m':>20s} slo/1k={r['cold_slo_per_1k']:7.4f}", flush=True)

    elif args.part == "lowfreq":
        trace = trace_lowfreq(seed=7, min_gap=5.0)
        n, mem = trace["n_apps"], trace["mem_mb"]
        print(f"低频子集: {n} apps, fully-warm={mem.sum()/1024:.0f} GB, "
              f"invocations={trace['counts'].sum():,.0f}", flush=True)
        for B_GB in (256, 512):
            B = B_GB * 1024.0
            for name, pol in [("Planner-LRU", PlannerLRU(n, mem, B)),
                              ("Planner-Popularity", PlannerPopularity(n, mem, B)),
                              ("BALM-JIT (hazard)", BALMJIT(n, mem, B)),
                              ("Oracle knapsack", OracleKnapsack(n, mem, B)),
                              ("TTL-20m", sim.FixedTTL(n, mem, ttl=20))]:
                r = sim.simulate(trace, pol, seed=0)
                r.update(part="lowfreq", policy=name, budget_gb=B_GB, seed=0,
                         full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"B={B_GB:5d} {name:>20s} cold/1k={r['cold_ratio_per_1k']:8.4f} "
                      f"pre/1k={r['prewarm_per_1k']:7.3f} inits/1k={r['total_inits_per_1k']:7.3f} "
                      f"cold/app/day={r['cold_per_app_day']:8.2f}", flush=True)

    elif args.part == "unpred":
        # 结构上不可预测 = 本次激活与上次激活的间隔 > G，候选窗口根本无法覆盖
        trace = sim.load_trace(seed=7)
        csr = trace["counts"].tocsr()
        res = {}
        for G in (60, 120, 360, 1440):
            ev = 0
            beyond = 0
            inv_tot = 0.0
            inv_beyond = 0.0
            for a in range(csr.shape[0]):
                s, e = csr.indptr[a], csr.indptr[a + 1]
                if e - s < 2:
                    continue
                t = csr.indices[s:e]
                c = csr.data[s:e]
                gaps = np.diff(t)
                ev += len(gaps)
                beyond += int((gaps > G).sum())
                inv_tot += float(c[1:].sum())
                inv_beyond += float(c[1:][gaps > G].sum())
            res[G] = dict(events=ev, beyond=beyond, share=beyond / max(ev, 1),
                          inv_share=inv_beyond / max(inv_tot, 1e-9))
            print(f"G={G:5d}: 超出窗口的激活 {beyond:,}/{ev:,} = {beyond/max(ev,1):.4%} "
                  f"(调用加权 {inv_beyond/max(inv_tot,1e-9):.4%})", flush=True)
        # 低频子集单独统计
        lf = trace_lowfreq(seed=7, min_gap=5.0)
        clf = lf["counts"].tocsr()
        ev = beyond = 0
        for a in range(clf.shape[0]):
            s, e = clf.indptr[a], clf.indptr[a + 1]
            if e - s < 2:
                continue
            gaps = np.diff(clf.indices[s:e])
            ev += len(gaps)
            beyond += int((gaps > 120).sum())
        print(f"低频子集 (mean gap>=5min), G=120: {beyond/max(ev,1):.3%} 的激活超出候选窗口",
              flush=True)
        pd.DataFrame([dict(part="unpred", G=g, **v) for g, v in res.items()]).to_csv(
            os.path.join(OUT, "unpredictable_share.csv"), index=False)

    elif args.part == "subminute":
        trace = sim.load_trace(seed=7)
        n, mem = trace["n_apps"], trace["mem_mb"]
        for B_GB in (768, 1024):
            B = B_GB * 1024.0
            for init_s in (0.5, 1.0, 2.0, 5.0):
                r = sim.simulate(trace, BALMJIT(n, mem, B), seed=0, init_s=init_s)
                r.update(part="subminute", policy="BALM-JIT (hazard)", budget_gb=B_GB,
                         seed=0, init_s=init_s, full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"B={B_GB} init={init_s}s cold/1k={r['cold_ratio_per_1k']:7.4f} "
                      f"+subminute={r['subminute_extra_per_1k']:7.4f} "
                      f"total={r['cold_plus_subminute_per_1k']:7.4f}", flush=True)
            for name, pol in [("Planner-LRU", PlannerLRU(n, mem, B)),
                              ("TTL-20m", sim.FixedTTL(n, mem, ttl=20))]:
                r = sim.simulate(trace, pol, seed=0, init_s=1.0)
                r.update(part="subminute", policy=name, budget_gb=B_GB, seed=0,
                         init_s=1.0, full_warm_gb=mem.sum() / 1024, n_apps=n)
                append(rows, r)
                print(f"B={B_GB} {name:>12s} cold/1k={r['cold_ratio_per_1k']:7.4f} "
                      f"+subminute={r['subminute_extra_per_1k']:7.4f}", flush=True)


if __name__ == "__main__":
    main()
