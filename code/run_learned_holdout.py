#!/usr/bin/env python3
"""
run_learned_holdout.py — 学习式预测基线的严格 holdout 评测。

协议（对所有方法一致）:
  · 前 5 天 (7200 分钟) 仅作为预热窗口: 所有策略的统计量在该窗口内更新,
    逻辑回归模型也只用这 5 天的样本训练;
  · 指标只在第 6-14 天 (12960 分钟) 上累计;
  · Learned / BALM / LRU / Popularity / Oracle / TTL-20m 使用完全相同的协议。

输出: results/learned_holdout.csv
"""
import os
import time
import numpy as np
import pandas as pd

import sim
from ml_planner import PlannerML, collect_training_data, train_model
from policies_plan import BALMJIT, OracleKnapsack, PlannerLRU, PlannerPopularity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
CSV = os.path.join(OUT, "learned_holdout.csv")
TRAIN_DAYS = 5
EVAL_START = TRAIN_DAYS * 1440


def main():
    trace = sim.load_trace(seed=7)
    n, mem = trace["n_apps"], trace["mem_mb"]
    print(f"trace: {n} apps, {trace['n_minutes']} minutes; "
          f"train = days 1-{TRAIN_DAYS}, eval = days {TRAIN_DAYS+1}-14", flush=True)

    t0 = time.time()
    X, y = collect_training_data(trace, train_minutes=EVAL_START, sample_stride=30,
                                 candidate_population=True, max_lag=120)
    model = train_model(X, y)   # 不加 class weight：训练分布=推理候选分布
    print(f"训练样本 {X.shape}, 正例率 {y.mean():.3f}, "
          f"in-sample acc {model.score(X, y):.4f} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for B_GB in (256, 768, 1024, 1536):
        B = B_GB * 1024.0
        pols = [
            ("Planner-Learned", PlannerML(n, mem, B, model=model, mean_s=trace["mean_s"])),
            ("BALM-JIT (hazard)", BALMJIT(n, mem, B)),
            ("Planner-LRU", PlannerLRU(n, mem, B)),
            ("Planner-Popularity", PlannerPopularity(n, mem, B)),
            ("Oracle knapsack", OracleKnapsack(n, mem, B)),
            ("TTL-20m", sim.FixedTTL(n, mem, ttl=20)),
        ]
        for name, pol in pols:
            t1 = time.time()
            r = sim.simulate(trace, pol, seed=0, eval_start=EVAL_START)
            r.update(policy=name, budget_gb=B_GB, seed=0, n_apps=n,
                     full_warm_gb=mem.sum() / 1024, train_days=TRAIN_DAYS,
                     eval_days=14 - TRAIN_DAYS)
            rows.append(r)
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(f"{name:>20s} B={B_GB:5d} cold/1k={r['cold_ratio_per_1k']:7.4f} "
                  f"mem={r['mean_prewarm_mem_gb']:7.1f} pre/1k={r['prewarm_per_1k']:7.3f} "
                  f"inits/1k={r['total_inits_per_1k']:7.3f} ({time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
