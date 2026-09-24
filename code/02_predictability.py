#!/usr/bin/env python3
"""
02_predictability.py — 负载可预测性分析（严格因果、全量向量化）。

对每个应用的每一次激活事件，仅用该事件之前的信息预测其发生分钟，统计容差命中率：
  median-k : 最近 k 个间隔的中位数（k=10，滑动窗口）
  last-gap : 最近一次间隔
并按应用典型激活间隔与触发器类型分层。
"""
import json
import os
import numpy as np

import sim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

TRIGGERS = ["http", "timer", "event", "queue", "storage", "orchestration", "others"]
STRATA = [(1, 2), (2, 10), (10, 60), (60, 1440), (1440, 10 ** 9)]
STRATA_NAMES = ["<=1 min", "2-9 min", "10-59 min", "1-24 h", ">24 h"]
TOL = [0, 1, 2, 5]
K = 10


def rolling_median(arr, k):
    """out[j] = median(arr[max(0, j-k+1) .. j])，向量化实现。"""
    m = len(arr)
    out = np.empty(m)
    if m == 0:
        return out
    for j in range(min(k - 1, m)):
        out[j] = np.median(arr[:j + 1])
    if m >= k:
        view = np.lib.stride_tricks.sliding_window_view(arr, k)
        out[k - 1:] = np.median(view, axis=1)
    return out


def analyze(ndays=14, napps=None, seed=7):
    trace = sim.load_trace(ndays=ndays, nsample_apps=napps, seed=seed)
    csr = trace["counts"].tocsr()
    n_apps = csr.shape[0]
    trig = trace["trigger"]

    preds = ["median-k", "last-gap"]
    hits = {p: {t: np.zeros(n_apps, dtype=np.int64) for t in TOL} for p in preds}
    events = np.zeros(n_apps, dtype=np.int64)
    inv_events = np.zeros(n_apps, dtype=np.float64)
    inv_hits = {p: {t: np.zeros(n_apps, dtype=np.float64) for t in TOL} for p in preds}
    mean_gap = np.zeros(n_apps, dtype=np.float64)

    for a in range(n_apps):
        s, e = csr.indptr[a], csr.indptr[a + 1]
        if e - s < 3:
            continue
        times = csr.indices[s:e].astype(np.int64)
        cnts = csr.data[s:e].astype(np.float64)
        gaps = np.diff(times)
        mean_gap[a] = gaps.mean()
        # 事件 i>=2 的预测值（只用到 i-1 及更早的间隔）
        med_run = rolling_median(gaps, K)
        # 预测事件 i 时只知道 times[1..i-1] 与 gaps[0..i-2]
        pred_med = times[1:-1] + med_run[:-1]
        pred_last = times[1:-1] + gaps[:-1]
        actual = times[2:]
        err_med = np.abs(pred_med - actual)
        err_last = np.abs(pred_last - actual)
        w = cnts[2:]
        events[a] = len(actual)
        inv_events[a] = w.sum()
        for t in TOL:
            hm = err_med <= t
            hl = err_last <= t
            hits["median-k"][t][a] = int(hm.sum())
            hits["last-gap"][t][a] = int(hl.sum())
            inv_hits["median-k"][t][a] = float(w[hm].sum())
            inv_hits["last-gap"][t][a] = float(w[hl].sum())

    tot = events.sum()
    result = {"n_apps": int(n_apps), "n_minutes": int(csr.shape[1]),
              "events": int(tot), "k": K, "tolerance": {}, "strata": {}, "trigger": {}}
    print(f"可预测性分析: {n_apps} apps, {csr.shape[1]} minutes, {tot:,} events")
    for p in preds:
        for t in TOL:
            hr = hits[p][t].sum() / max(tot, 1)
            ihr = inv_hits[p][t].sum() / max(inv_events.sum(), 1)
            result["tolerance"][f"{p}|{t}"] = dict(event_hit_rate=float(hr),
                                                   invocation_hit_rate=float(ihr))
            print(f"  {p:>9s} ±{t}min: 事件命中率 {hr:6.1%} | 调用加权 {ihr:6.1%}")

    print("\n按典型激活间隔分层 (median-k, ±1min):")
    for (lo, hi), name in zip(STRATA, STRATA_NAMES):
        m = (mean_gap >= lo) & (mean_gap < hi)
        ev = events[m].sum()
        if ev == 0:
            continue
        hr = hits["median-k"][1][m].sum() / ev
        result["strata"][name] = dict(apps=int(m.sum()), events=int(ev),
                                      hit_rate_1min=float(hr))
        print(f"  {name:>10s}: 应用 {int(m.sum()):>6d}, 事件 {int(ev):>12,}, "
              f"±1min 命中率 {hr:6.1%}")

    print("\n按触发器 (median-k, ±1min):")
    for ti, tname in enumerate(TRIGGERS):
        m = (trig == ti) & (events > 0)
        ev = events[m].sum()
        if ev == 0:
            continue
        hr = hits["median-k"][1][m].sum() / ev
        result["trigger"][tname] = dict(apps=int(m.sum()), events=int(ev),
                                        hit_rate_1min=float(hr))
        print(f"  {tname:>14s}: 应用 {int(m.sum()):>6d}, 事件 {int(ev):>12,}, "
              f"±1min 命中率 {hr:6.1%}")

    with open(os.path.join(OUT, "predictability.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n已写出 results/predictability.json")
    return result


if __name__ == "__main__":
    import sys
    nd = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    analyze(ndays=nd, seed=7)
