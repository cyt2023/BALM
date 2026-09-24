#!/usr/bin/env python3
"""run_ml_baseline.py — 训练学习式预测器并作为 prewarming baseline 参与评估。

训练窗口 = trace 前 5 天（7200 分钟），评估 = 全部 14 天（模型冻结）。
输出: results/revision_ml.csv
"""
import os
import time
import numpy as np
import pandas as pd

import sim
from ml_planner import PlannerML, collect_training_data, train_model
from policies_plan import BALMJIT, OracleKnapsack, PlannerLRU

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
CSV = os.path.join(OUT, "revision_ml.csv")


def main():
    trace = sim.load_trace(seed=7)
    n, mem = trace["n_apps"], trace["mem_mb"]
    print(f"trace: {n} apps, {trace['n_minutes']} minutes", flush=True)

    t0 = time.time()
    X, y = collect_training_data(trace, train_minutes=5 * 1440, sample_stride=15,
                                 neg_per_min=400)
    print(f"训练样本: X={X.shape}, 正例率={y.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
    model = train_model(X, y)
    acc = model.score(X, y)
    print(f"逻辑回归训练完成 in-sample accuracy={acc:.4f}", flush=True)
    np.save(os.path.join(OUT, "ml_model_coef.npy"), model.coef_)

    rows = pd.read_csv(CSV).to_dict("records") if os.path.exists(CSV) else []
    done = {(r["policy"], r["budget_gb"]) for r in rows}
    for B_GB in (256, 768, 1024, 1536):
        B = B_GB * 1024.0
        pols = [("Planner-Learned", PlannerML(n, mem, B, model=model,
                                              mean_s=trace["mean_s"])),
                ("BALM-JIT (hazard)", BALMJIT(n, mem, B)),
                ("Planner-LRU", PlannerLRU(n, mem, B)),
                ("Oracle knapsack", OracleKnapsack(n, mem, B))]
        for name, pol in pols:
            if (name, B_GB) in done:
                continue
            t1 = time.time()
            r = sim.simulate(trace, pol, seed=0)
            r.update(policy=name, budget_gb=B_GB, seed=0,
                     full_warm_gb=mem.sum() / 1024, n_apps=n)
            rows.append(r)
            pd.DataFrame(rows).to_csv(CSV, index=False)
            print(f"{name:>20s} B={B_GB:5d} cold/1k={r['cold_ratio_per_1k']:7.4f} "
                  f"pre/1k={r['prewarm_per_1k']:7.3f} inits/1k={r['total_inits_per_1k']:7.3f} "
                  f"({time.time()-t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
