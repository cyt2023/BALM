#!/usr/bin/env python3
"""03_characterize.py — 负载特征统计，输出 results/characterization.json 与绘图用数据。"""
import json
import os
import numpy as np
import pandas as pd

import sim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
TRIGGERS = ["http", "timer", "event", "queue", "storage", "orchestration", "others"]


def main(ndays=14, seed=7):
    trace = sim.load_trace(ndays=ndays, seed=seed)
    c = trace["counts"].tocsc()
    n_apps, n_min = c.shape
    mem = trace["mem_mb"]
    inv = np.asarray(c.sum(axis=0)).ravel()
    active = np.diff(c.indptr)            # 每分钟活跃应用数

    # 间隔直方图
    csr = c.tocsr()
    gaps = []
    for a in range(0, n_apps):
        s, e = csr.indptr[a], csr.indptr[a + 1]
        if e - s >= 2:
            gaps.append(np.diff(csr.indices[s:e]))
    gaps = np.concatenate(gaps)

    trig_counts = {TRIGGERS[i]: int((trace["trigger"] == i).sum()) for i in range(7)}

    stats = dict(
        n_apps=int(n_apps), n_minutes=int(n_min), ndays=ndays,
        total_invocations=float(c.sum()),
        mem_mb=dict(mean=float(mem.mean()), median=float(np.median(mem)),
                    p95=float(np.percentile(mem, 95)), p99=float(np.percentile(mem, 99)),
                    max=float(mem.max()), fully_warm_gb=float(mem.sum() / 1024)),
        active_apps_per_minute=dict(mean=float(active.mean()),
                                    p95=float(np.percentile(active, 95)),
                                    max=int(active.max())),
        concurrent_active_mem_gb=dict(mean=float((mem[np.nonzero(c[:, t].toarray().ravel())[0]].sum()
                                                  for t in []) or 0) if False else None),
        invocations_per_active_app_minute=float(c.sum() / active.sum()),
        gap_minutes=dict(mean=float(gaps.mean()), median=float(np.median(gaps)),
                         p95=float(np.percentile(gaps, 95)),
                         share_eq_1=float((gaps == 1).mean()),
                         share_le_5=float((gaps <= 5).mean()),
                         share_ge_60=float((gaps >= 60).mean())),
        trigger_apps=trig_counts,
    )

    # 每分钟并发活跃集的内存（用于说明预算数量级）
    mem_per_min = np.zeros(n_min)
    indptr, indices = c.indptr, c.indices
    for t in range(n_min):
        s, e = indptr[t], indptr[t + 1]
        if e > s:
            mem_per_min[t] = mem[indices[s:e]].sum()
    stats["concurrent_active_mem_gb"] = dict(
        mean=float(mem_per_min.mean() / 1024), p95=float(np.percentile(mem_per_min, 95) / 1024),
        max=float(mem_per_min.max() / 1024))

    np.savez_compressed(os.path.join(OUT, "char_data.npz"),
                        mem_mb=mem.astype(np.float32),
                        gap_hist=np.bincount(np.clip(gaps, 0, 1440), minlength=1441),
                        active_per_min=active.astype(np.int32),
                        inv_per_min=inv.astype(np.float64),
                        mem_per_min=mem_per_min.astype(np.float32),
                        trigger=trace["trigger"])

    with open(os.path.join(OUT, "characterization.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
