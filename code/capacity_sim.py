#!/usr/bin/env python3
"""capacity_sim.py — 单一总内存上限下的策略比较（本次补实验的核心）。

与 sim.simulate 的区别：
  · 引入硬总容量 C（MB），同时约束“预选热集合”和“本分钟按需启动的容器”，
    任意时刻的模型内总占用不得超过 C；
  · 请求顺序：同一分钟内各应用的请求按固定随机种子的顺序处理。为免逐条枚举
    （一分钟可达数十万次调用），使用指数时钟等价性质：对活跃应用 a 取 E_a~Exp(1)，
    按 E_a / count_a 排序，等价于对请求多重集做均匀随机排列后各应用的首次出现顺序；
  · 容量不足时的统一规则：
      1) 只能驱逐“空闲的预选热容器”（本分钟尚未被服务到的 warm 容器）；
         本分钟已服务过的容器（正在执行）永不驱逐；
      2) 驱逐对象按 LRU（last_active 最旧优先，应用 id 升序打破平局），
         只用历史信息，绝不使用本分钟的未来请求；
      3) 驱逐后仍放不下，则该应用本分钟全部请求记为未服务（rejected），
         如实计入完成率；不凭空增加内存、不静默丢弃。
  · 保留原论文边界：分钟粒度、每应用每分钟至多一次冷启动、不模拟并发扩容。
  · Oracle 仍是非因果上界，只作参照。
"""

from __future__ import annotations

import numpy as np

from sim import COLD_MEDIAN_S, SLO_SEC, draw_cold_penalties


def simulate_capacity(trace, policy, total_capacity_mb, slo=SLO_SEC, seed=0,
                      cold_median_s=COLD_MEDIAN_S, cold_penalty_mode="memory",
                      eval_start=0):
    """在硬总容量 C 下回放轨迹，返回指标字典。

    额外指标：rejected_requests / served_requests / service_completion_rate /
    capacity_evictions / blocked_minutes / mean,p95,max_total_mem_gb /
    capacity_violations（应恒为 0）。
    """
    counts = trace["counts"]
    mem = trace["mem_mb"]
    n_apps = trace["n_apps"]
    n_min = trace["n_minutes"]
    C = float(total_capacity_mb)

    rng = np.random.default_rng(seed)
    cold_pen = draw_cold_penalties(mem, rng, median_s=cold_median_s,
                                   mode=cold_penalty_mode)
    if hasattr(policy, "set_aux"):
        policy.set_aux(trace["mean_s"], cold_pen)

    nbins = 220
    edges = np.logspace(np.log10(0.001), np.log10(60.0), nbins + 1)
    hist = np.zeros(nbins, dtype=np.float64)
    q_s = trace["q_s"]
    q_w = np.diff(np.concatenate([[0.0], trace["q_w"].astype(np.float64)]))
    nq = q_s.shape[1]
    bin_idx = [np.clip(np.digitize(q_s[:, j], edges) - 1, 0, nbins - 1) for j in range(nq)]
    cold_bin = [np.clip(np.digitize(q_s[:, j] + cold_pen, edges) - 1, 0, nbins - 1)
                for j in range(nq)]

    indptr, indices, data = counts.indptr, counts.indices, counts.data
    warm_until = policy.warm_until
    EMPTY = np.empty(0, dtype=np.int64)

    occ_minutes = np.zeros(n_min, dtype=np.float64)
    warm_minutes = np.zeros(n_min, dtype=np.float64)
    cold_events = 0
    cold_slo = 0.0
    served = 0.0
    rejected = 0.0
    cold_ms_sum = 0.0
    latency_ms_sum = 0.0
    prewarm_attempts = 0
    capacity_evictions = 0
    blocked_minutes = 0
    violations = 0
    resident_prev = np.zeros(n_apps, dtype=bool)
    last_seen = np.full(n_apps, -10**9, dtype=np.int64)
    cand_sizes = []
    active_flag = np.zeros(n_apps, dtype=bool)

    for t in range(n_min):
        counting = t >= eval_start
        s, e = indptr[t], indptr[t + 1]
        if s == e:
            apps = EMPTY
            cnt = np.empty(0, dtype=np.float64)
        else:
            apps = indices[s:e]
            cnt = data[s:e].astype(np.float64)

        replan_every = getattr(policy, "delta", 0) or 1
        if t % replan_every == 0:
            policy.replan(t, apps, cnt)
            k = getattr(policy, "last_cand_size", None)
            if k:
                cand_sizes.append(k)

        warm_mask = warm_until >= t
        warm_mem = float(mem[warm_mask].sum())
        # 分钟开始时的预驱逐：若策略的预选热集合本身超过硬容量 C（策略只知道自己
        # 的分配预算，不一定知道 C），按与请求阶段相同的 LRU 规则驱逐空闲热容器，
        # 直到占用 ≤ C。这一步对所有策略一致，且只用历史信息。
        if warm_mem > C:
            idle = np.nonzero(warm_mask)[0]
            while warm_mem > C and len(idle):
                key = np.lexsort((idle, last_seen[idle]))
                v = int(idle[key[0]])
                warm_until[v] = t - 1
                warm_mem -= float(mem[v])
                if counting:
                    capacity_evictions += 1
                idle = np.nonzero(warm_until >= t)[0]
            warm_mask = warm_until >= t
        warm_set_mem = warm_mem
        active_flag[:] = False
        active_flag[apps] = True
        newly = warm_mask & ~resident_prev
        n_new = int(newly.sum())
        if counting and n_new:
            prewarm_attempts += n_new

        # 所有尚未被本分钟请求使用的热容器都可能成为驱逐对象。不能用
        # active_flag 提前挑选本分钟最终不活跃的应用，否则泄漏未来请求。
        # 只根据已观察到的 last_seen 排序，同分时按应用 id 升序。
        eviction_order = np.nonzero(warm_mask)[0]
        if len(eviction_order):
            eviction_order = eviction_order[
                np.lexsort((eviction_order, last_seen[eviction_order]))
            ]
        victim_ptr = 0

        idle_warm = warm_mask.copy()
        occ = warm_mem
        served_mask = np.zeros(n_apps, dtype=bool)
        cold_apps_t = []
        rejected_here = 0
        need_mask = active_flag & ~warm_mask
        fast_ok = bool(len(apps)) and (warm_set_mem + float(mem[need_mask].sum()) <= C)
        if fast_ok:
            # 无冲突：所有活跃应用都能被服务，无需驱逐/拒绝（与慢路径结果等价）
            cold_apps_t = list(np.nonzero(need_mask)[0])
            occ = warm_set_mem + float(mem[need_mask].sum())
            served_mask = active_flag.copy()
            warm_until[need_mask] = t
        elif len(apps):
            r = np.random.default_rng((seed * 1_000_003 + t) % (2**32))
            u = r.exponential(size=len(apps))
            order = np.argsort(u / np.maximum(cnt, 1e-12), kind="stable")
            for idx in order:
                a = int(apps[idx])
                if warm_until[a] >= t:
                    served_mask[a] = True
                    idle_warm[a] = False
                    continue
                need = float(mem[a])
                if occ + need <= C:
                    warm_until[a] = t
                    occ += need
                    cold_apps_t.append(a)
                    served_mask[a] = True
                    continue
                need_mb = need - (C - occ)
                while need_mb > 0:
                    v = -1
                    while victim_ptr < len(eviction_order):
                        cand_v = int(eviction_order[victim_ptr]); victim_ptr += 1
                        if idle_warm[cand_v] and warm_until[cand_v] >= t:
                            v = cand_v; break
                    if v < 0:
                        break
                    warm_until[v] = t - 1
                    idle_warm[v] = False
                    occ -= float(mem[v])
                    need_mb -= float(mem[v])
                    if counting:
                        capacity_evictions += 1
                if occ + need <= C:
                    warm_until[a] = t
                    occ += need
                    cold_apps_t.append(a)
                    served_mask[a] = True
                else:
                    if counting:
                        rejected += float(cnt[idx])
                    rejected_here += 1

        if rejected_here and counting:
            blocked_minutes += 1

        cold_apps_t = np.asarray(cold_apps_t, dtype=np.int64)
        n_cold = len(cold_apps_t)
        served_cnt = cnt
        if rejected_here and len(apps):
            served_cnt = np.where(~served_mask[apps], 0.0, cnt)

        if counting:
            cold_events += n_cold
            served += float(served_cnt.sum())
            if n_cold:
                cold_ms_sum += float(cold_pen[cold_apps_t].sum())
                cold_slo += float(((trace["mean_s"][cold_apps_t] + cold_pen[cold_apps_t]) > slo).sum())
            for j in range(nq):
                hist += np.bincount(bin_idx[j][apps], weights=served_cnt * q_w[j], minlength=nbins)
                if n_cold:
                    hist += np.bincount(cold_bin[j][cold_apps_t], weights=np.full(n_cold, q_w[j]),
                                        minlength=nbins)
                    hist -= np.bincount(bin_idx[j][cold_apps_t], weights=np.full(n_cold, q_w[j]),
                                        minlength=nbins)
            latency_ms_sum += float((served_cnt * trace["mean_ms"][apps]).sum())
            occ_minutes[t] = occ
            warm_minutes[t] = warm_set_mem
        if occ > C + 1e-6:
            violations += 1

        last_seen[apps] = t
        resident_prev[:] = warm_until >= t
        policy.tick(t, apps)
        policy.on_activity(t, apps, cnt)

    ev = occ_minutes[eval_start:]
    wv = warm_minutes[eval_start:]
    tot_w = hist.sum()
    cum = np.cumsum(hist) / max(tot_w, 1.0)

    def qtl(p):
        i = int(np.searchsorted(cum, p))
        return float(edges[min(i, nbins - 1)])

    request_total = served + rejected
    return dict(
        policy=policy.name,
        total_capacity_gb=C / 1024.0,
        request_seed=seed,
        cold_events=int(cold_events),
        served_requests=float(served),
        rejected_requests=float(rejected),
        service_completion_rate=float(served / request_total) if request_total else float("nan"),
        cold_ratio_per_1k=(cold_events / served * 1000.0) if served else float("nan"),
        cold_ratio_per_1k_all_requests=(cold_events / request_total * 1000.0) if request_total else float("nan"),
        cold_slo_per_1k=(cold_slo / served * 1000.0) if served else float("nan"),
        prewarm_attempts=int(prewarm_attempts),
        total_inits=int(cold_events + prewarm_attempts),
        capacity_evictions=int(capacity_evictions),
        blocked_minutes=int(blocked_minutes),
        mean_total_mem_gb=float(ev.mean()) / 1024.0,
        p95_total_mem_gb=float(np.percentile(ev, 95)) / 1024.0,
        max_total_mem_gb=float(ev.max()) / 1024.0,
        mean_warm_set_mem_gb=float(wv.mean()) / 1024.0,
        max_warm_set_mem_gb=float(wv.max()) / 1024.0,
        capacity_violations=int(violations),
        mean_latency_s=(latency_ms_sum + cold_ms_sum) / max(served, 1e-9) / 1000.0,
        p95_latency_s=qtl(0.95),
        p99_latency_s=qtl(0.99),
        slo_violation_rate=float(hist[(edges[:-1] >= slo)].sum() / max(tot_w, 1)),
        cand_k_mean=float(np.mean(cand_sizes)) if cand_sizes else float("nan"),
    )
