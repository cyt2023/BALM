#!/usr/bin/env python3
"""run_capacity.py — 单一总内存上限（hard total cap）下的策略对比实验。

协议（对所有策略一致）:
  · 硬总容量 C 同时约束预选热集合与本分钟按需启动的容器；
  · 每个策略的内部分配预算也设为 C（模拟器另有兜底预驱逐，保证占用不超 C）；
  · 同一分钟内的请求顺序由固定随机种子决定，各策略共享同一顺序；
  · 容量不足时：只能驱逐空闲的预选热容器（LRU，历史信息），执行中的容器不可驱逐；
    仍放不下则该应用本分钟请求记为未服务（rejected），并计入完成率。

用法:
  python3 run_capacity.py --ndays 2 --napps 3000 --tag pilot
  python3 run_capacity.py --tag full            # 完整 14 天 / 全部应用（按资源）
输出: results/capacity_<tag>.csv
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import sim
from capacity_sim import simulate_capacity
from ml_planner import PlannerML, collect_training_data, train_model
from policies_plan import BALMJIT, OracleKnapsack, PlannerLRU, PlannerPopularity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

CAP_FRACTIONS = [0.6, 0.8, 1.0, 1.2]
SEEDS = [0, 1, 2]
TRAIN_DAYS_FOR_LEARNED = 1          # pilot：用首日训练；完整规模时用 5 天
PROTOCOL_VERSION = "causal-eviction-eval-window-v2"


def build_policies(n, mem, budget_mb, model=None, mean_s=None):
    """返回 [(family, name, factory)]；family 用于区分预测型/非预测型/租约型。"""
    pols = [
        ("predictive-online", BALMJIT(n, mem, budget_mb)),
        ("non-predictive", PlannerLRU(n, mem, budget_mb)),
        ("non-predictive", PlannerPopularity(n, mem, budget_mb)),
        ("non-causal-upper-bound", OracleKnapsack(n, mem, budget_mb)),
        ("lease-reference", sim.FixedTTL(n, mem, ttl=20)),
        ("lease-reference", sim.HybridHistogram(n, mem)),
    ]
    if model is not None:
        pols.insert(0, ("predictive-learned",
                        PlannerML(n, mem, budget_mb, model=model, mean_s=mean_s)))
    return pols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndays", type=int, default=None)
    ap.add_argument("--napps", type=int, default=None)
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--fractions", type=str, default="0.6,0.8,1.0,1.2")
    ap.add_argument("--learned-seeds", type=str, default="0")
    ap.add_argument("--cheap-seeds", type=str, default="0,1,2")
    ap.add_argument("--train-days", type=int, default=1)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    learned_seeds = [int(s) for s in args.learned_seeds.split(",")]
    cheap_seeds = [int(s) for s in args.cheap_seeds.split(",")]
    fractions = [float(x) for x in args.fractions.split(",")]

    rows = []
    done = set()
    csv_path = os.path.join(OUT, f"capacity_{args.tag}.csv")
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        if len(prev) and ("protocol_version" not in prev or
                          not prev["protocol_version"].eq(PROTOCOL_VERSION).all()):
            raise ValueError(
                f"{csv_path} contains results from an older capacity protocol. "
                "Use a new --tag (for example causal-full) and rerun; "
                "do not mix or resume the published pre-audit numbers."
            )
        rows = prev.to_dict("records")
        done = {(r["policy"], round(float(r["total_capacity_gb"]), 3), int(r["request_seed"]))
                for r in rows}
        print(f"resume: 已有 {len(rows)} 行，跳过重复配置", flush=True)

    trace = sim.load_trace(ndays=args.ndays, nsample_apps=args.napps, seed=7)
    n, mem = trace["n_apps"], trace["mem_mb"]
    n_min = trace["n_minutes"]
    print(f"trace: {n} apps, {n_min} minutes ({n_min/1440:.1f} days), "
          f"fully-warm={mem.sum()/1024:.0f} GB", flush=True)

    # 训练学习式基线（因果：只用前若干天）
    train_minutes = min(args.train_days * 1440, max(n_min - 1440, 1440))
    # stride 30 与论文中的学习基线一致（240 个抽样分钟、约 290 万行）；stride 15 会
    # 让 lbfgs 拟合耗时增加一个数量级且不改变结论
    X, y = collect_training_data(trace, train_minutes=train_minutes, sample_stride=30,
                                 candidate_population=True, max_lag=120, verbose=True)
    model = train_model(X, y)
    print(f"learned baseline trained on first {train_minutes/1440:.1f} day(s), "
          f"prevalence={y.mean():.4f}", flush=True)

    # 用一次"无硬上限"的 BALM 运行确定容量尺度（策略预算取 fully-warm 的 25%）
    ref_budget_mb = float(np.clip(mem.sum() * 0.25, 32 * 1024, 1024 * 1024))
    t0 = time.time()
    ref = sim.simulate(trace, BALMJIT(n, mem, ref_budget_mb), seed=0)
    ref_total = ref["mean_total_mem_gb"]
    print(f"reference (no hard cap, budget={ref_budget_mb/1024:.0f} GB): "
          f"cold/1k={ref['cold_ratio_per_1k']:.3f} mean_total={ref_total:.1f} GB "
          f"({time.time()-t0:.0f}s)", flush=True)

    for frac in fractions:
        C = ref_total * 1024.0 * frac          # 硬总容量（MB）
        for seed in seeds:
            for family, pol in build_policies(n, mem, C, model=model,
                                              mean_s=trace["mean_s"]):
                if family == "predictive-learned" and seed not in learned_seeds:
                    continue
                if family in ("predictive-online", "non-predictive") and seed not in cheap_seeds:
                    continue
                if family in ("non-causal-upper-bound", "lease-reference") and seed != 0:
                    continue
                if (pol.name, round(C / 1024.0, 3), seed) in done:
                    continue
                t0 = time.time()
                try:
                    r = simulate_capacity(trace, pol, total_capacity_mb=C, seed=seed,
                                          eval_start=train_minutes)
                except Exception as e:          # noqa: BLE001
                    print(f"  !! {pol.name} frac={frac} seed={seed}: {e}", flush=True)
                    raise
                r.update(family=family, capacity_frac_of_reference=frac,
                         ndays=(n_min / 1440), n_apps=n, eval_start=train_minutes,
                         protocol_version=PROTOCOL_VERSION)
                rows.append(r)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"  C={C/1024:7.1f}GB ({frac:>4.1f}x) seed={seed} {pol.name:22s} "
                      f"served={r['service_completion_rate']*100:6.2f}% "
                      f"cold/1k={r['cold_ratio_per_1k']:8.3f} "
                      f"rej/1k={r['rejected_requests']/max(r['served_requests']+r['rejected_requests'],1)*1000:8.2f} "
                      f"evict={r['capacity_evictions']:7d} "
                      f"maxmem={r['max_total_mem_gb']:7.1f} viol={r['capacity_violations']} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"capacity_{args.tag}.csv"), index=False)
    print(f"\n-> results/capacity_{args.tag}.csv  ({len(df)} rows)")


if __name__ == "__main__":
    main()
