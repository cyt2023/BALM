#!/usr/bin/env python3
"""快速自检：在小规模子集上验证仿真器的正确性与性能。"""
import time
import numpy as np
import sim


def main():
    t0 = time.time()
    trace = sim.load_trace(ndays=1, nsample_apps=3000, seed=1)
    print(f"trace: {trace['n_apps']} apps x {trace['n_minutes']} min "
          f"({time.time()-t0:.1f}s), invocations={trace['counts'].sum():,.0f}")

    n = trace["n_apps"]
    mem = trace["mem_mb"]
    budget = 20 * 1024.0        # 20 GB 预算（子集）

    cases = [
        ("Scale-to-zero", sim.NoKeepAlive(n, mem)),
        ("TTL-1m", sim.FixedTTL(n, mem, ttl=1)),
        ("TTL-10m", sim.FixedTTL(n, mem, ttl=10)),
        ("Historic-TTL", sim.HybridHistogram(n, mem)),
        ("Popularity", sim.TopKStatic(n, mem, budget)),
        ("LRU", sim.LRUProfit(n, mem, budget)),
        ("BALM", sim.BALM(n, mem, budget)),
    ]
    for name, pol in cases:
        t1 = time.time()
        res = sim.simulate(trace, pol, seed=0)
        print(f"{name:>22s} | cold/1k={res['cold_ratio_per_1k']:8.1f} "
              f"| cold_minute={res['cold_minute_ratio']:6.3f} "
              f"| p95={res['p95_latency_s']:6.3f}s "
              f"| slo_viol={res['slo_violation_rate']:6.3f} "
              f"| mem={res['mean_prewarm_mem_gb']:7.2f}GB "
              f"| {time.time()-t1:5.1f}s")
        if name == "Scale-to-zero":
            assert abs(res["cold_minute_ratio"] - 1.0) < 1e-6, \
                "scale-to-zero 下每个活跃应用-分钟都应是冷启动"
        if name == "TTL-10m":
            assert res["cold_ratio_per_1k"] < 50, "10 分钟 TTL 应显著降低冷启动"

    print("\n自检通过。")


if __name__ == "__main__":
    main()
