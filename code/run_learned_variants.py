#!/usr/bin/env python3
"""
run_learned_variants.py — 学习式预测基线的采样敏感性实验。

两种训练分钟抽样方式（其余完全相同：候选集=推理候选集、L2/LBFGS、无 class weight）：
  random : 随机抽 240 个 burn-in 分钟（默认，无混叠）
  stride : 每 30 分钟取一次（t % 30 == 0）——会与 workload 的 3/5/30/60 分钟周期性混叠

评估协议：days 1-5 训练/burn-in，days 6-14 评测，与其它方法一致。
输出：results/learned_variants.csv
"""
import argparse, os, time
import numpy as np
import pandas as pd

import sim
from ml_planner import PlannerML, collect_training_data, train_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
CSV = os.path.join(OUT, "learned_variants.csv")
TRAIN_DAYS = 5
EVAL_START = TRAIN_DAYS * 1440


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling", choices=["random", "stride"], required=True)
    ap.add_argument("--budgets", default="256,768,1024,1536")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    trace = sim.load_trace(seed=7)
    n, mem = trace["n_apps"], trace["mem_mb"]
    kw = dict(candidate_population=True, max_lag=120, seed=0, verbose=True)
    if args.sampling == "random":
        kw.update(minute_sampling="random", n_sample_minutes=240)
    else:
        kw.update(minute_sampling="stride", sample_stride=30, offset=0)
    X, y = collect_training_data(trace, train_minutes=EVAL_START, **kw)
    model = train_model(X, y)
    name = "Planner-Learned" if args.sampling == "random" else "Planner-Learned (stride-30 sampling)"
    print(f"training: {X.shape}, prevalence={y.mean():.4f}, mean_pred="
          f"{model.predict_proba(X)[:,1].mean():.4f}", flush=True)

    rows = pd.read_csv(CSV).to_dict("records") if os.path.exists(CSV) else []
    done = {(r["policy"], r["budget_gb"]) for r in rows}
    for B_GB in budgets:
        if (name, B_GB) in done:
            continue
        t0 = time.time()
        r = sim.simulate(trace, PlannerML(n, mem, B_GB * 1024.0, model=model,
                                          mean_s=trace["mean_s"]), seed=0,
                         eval_start=EVAL_START)
        r.update(policy=name, budget_gb=B_GB, seed=0, n_apps=n,
                 full_warm_gb=mem.sum() / 1024, sampling=args.sampling,
                 train_rows=len(y), train_prevalence=float(y.mean()))
        rows.append(r)
        pd.DataFrame(rows).to_csv(CSV, index=False)
        print(f"{name:36s} B={B_GB:5d} cold/1k={r['cold_ratio_per_1k']:7.4f} "
              f"Brier={r['predictor_brier']:.4f} AUROC={r['predictor_auroc']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
