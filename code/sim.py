#!/usr/bin/env python3
"""
sim.py — 多租户 Serverless 平台"预热内存预算"下的容器保活策略仿真器。

模型（与论文 §3 一致）:
  · 时间离散为 1 分钟，与 trace 分辨率一致 (20,160 分钟 = 14 天)。
  · 应用 a 在分钟 t 的调用次数 c[a,t] 取自真实 trace。
  · 每个应用有实测内存占用 m[a] (MB) 与调用加权耗时分布 D[a]。
  · 保活租约: 调用发生在分钟 s 时设置 warm_until[a] = s + L(a)。
    若某分钟首个调用发生时 warm_until[a] < t，则该分钟产生一次冷启动。
  · 冷启动额外时延 T_cold(a) ~ LogNormal(median, sigma)，按内存规模缩放。
  · 策略只看历史信息（因果），不偷看未来。

输出指标:
  cold_events, cold_ratio_per_1k, mean_latency_s, p95/p99_latency_s,
  slo_violation_rate, mean_prewarm_mem_gb, p95_prewarm_mem_gb
"""

from __future__ import annotations

import os
import numpy as np
from scipy import sparse

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived", "trace.npz")

SLO_SEC = 1.0              # 默认 SLO：端到端时延 1 s
COLD_MEDIAN_S = 1.0        # 冷启动中位时延
COLD_SIGMA = 0.6           # 冷启动对数正态形状参数
COLD_CLIP_S = (0.1, 15.0)
MIN_DUR_S = 0.001


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_trace(ndays: int | None = None, nsample_apps: int | None = None, seed: int = 0):
    z = np.load(DATA, allow_pickle=False)
    shape = tuple(z["counts_shape"])
    counts = sparse.csr_matrix(
        (z["counts_data"], z["counts_indices"], z["counts_indptr"]), shape=shape
    )
    total_minutes = int(z["total_minutes"][0])
    mpd = int(z["minutes_per_day"][0])
    keep_minutes = total_minutes if ndays is None else ndays * mpd
    counts = counts[:, :keep_minutes].tocsc()

    apps = np.arange(shape[0])
    if nsample_apps is not None and nsample_apps < shape[0]:
        rng = np.random.default_rng(seed)
        apps = np.sort(rng.choice(shape[0], size=nsample_apps, replace=False))
        counts = counts[apps, :].tocsc()

    trace = dict(
        counts=counts,
        mem_mb=np.nan_to_num(z["app_mem_mb"][apps], nan=128.0).astype(np.float32),
        mean_ms=np.nan_to_num(z["app_mean_ms"][apps], nan=200.0).astype(np.float32),
        q_ms=np.nan_to_num(z["app_q_ms"][apps], nan=200.0).astype(np.float32),
        q_w=z["app_q_w"].astype(np.float32),
        trigger=z["app_trigger"][apps],
        n_minutes=counts.shape[1],
        n_apps=counts.shape[0],
    )
    trace["mean_s"] = np.clip(trace["mean_ms"] / 1000.0, MIN_DUR_S, None)
    trace["q_s"] = np.clip(trace["q_ms"] / 1000.0, MIN_DUR_S, None)
    return trace


def draw_cold_penalties(mem_mb: np.ndarray, rng: np.random.Generator,
                        median_s: float = COLD_MEDIAN_S,
                        sigma: float = COLD_SIGMA,
                        mode: str = "memory") -> np.ndarray:
    """冷启动时延模型。

    mode = "memory"      : 中位数随内存规模轻微上升（大镜像初始化更慢，默认）
    mode = "constant"    : 所有应用相同，等于 median_s（与内存无关的常量）
    mode = "independent" : 与内存相互独立的对数正态
    """
    if mode == "memory":
        scale = np.clip((mem_mb / 128.0) ** 0.15, 0.6, 2.0)
    elif mode == "constant":
        scale = np.ones_like(mem_mb, dtype=np.float64)
    elif mode == "independent":
        scale = np.ones_like(mem_mb, dtype=np.float64)
    else:
        raise ValueError(f"unknown cold-penalty mode: {mode}")
    mu = np.log(median_s * scale)
    t = rng.lognormal(mean=mu, sigma=sigma)
    return np.clip(t, *COLD_CLIP_S).astype(np.float32)


# --------------------------------------------------------------------------
# 策略
# --------------------------------------------------------------------------
class BasePolicy:
    name = "base"
    budget_limited = False

    def __init__(self, n_apps, mem_mb, **kw):
        self.n_apps = n_apps
        self.mem_mb = mem_mb
        # 在线统计
        self.last_active = np.full(n_apps, -10**9, dtype=np.int64)
        self.total_inv = np.zeros(n_apps, dtype=np.float64)
        self.ewma_rate = np.zeros(n_apps, dtype=np.float64)
        self.n_obs = np.zeros(n_apps, dtype=np.int32)
        # 间隔直方图（log2 分箱，1..1024 分钟）
        self.gap_hist = np.zeros((n_apps, 11), dtype=np.int32)
        self.warm_until = np.full(n_apps, -1, dtype=np.int64)
        self.alpha = 0.3
        self.lease_default = 0

    # 子类覆盖：在每个重规划点决定 lease
    def replan(self, t, active_apps, counts):
        pass

    def tick(self, t, active_apps):
        """每分钟调用一次（无论该分钟是否有调用），用于维护滑动窗口统计。"""
        return

    def on_activity(self, t, active_apps, counts):
        """调用发生后更新统计与租约。"""
        prev = self.last_active[active_apps]
        gaps = t - prev
        seen = gaps > 0
        if seen.any():
            idx = np.clip(np.log2(np.maximum(gaps[seen], 1)).astype(np.int32), 0, 10)
            np.add.at(self.gap_hist, (active_apps[seen], idx), 1)
        self.last_active[active_apps] = t
        self.total_inv[active_apps] += counts
        self.ewma_rate[active_apps] = (
            self.alpha * counts + (1 - self.alpha) * self.ewma_rate[active_apps]
        )
        self.n_obs[active_apps] += 1

    def reuse_prob(self, apps, window_min):
        """P(gap <= window)，来自应用自身间隔直方图；样本不足时回退全局。"""
        b = np.clip(np.log2(max(window_min, 1)), 0, 10)
        k = int(np.floor(b)) + 1
        h = self.gap_hist[apps, :k].sum(axis=1)
        tot = self.gap_hist[apps].sum(axis=1)
        p = np.divide(h, np.maximum(tot, 1), dtype=np.float64)
        return p, tot

    def lease_grant(self, apps, t, lengths):
        np.maximum.at(self.warm_until, apps, t + lengths)


class NoKeepAlive(BasePolicy):
    name = "Scale-to-zero"

    def on_activity(self, t, active_apps, counts):
        super().on_activity(t, active_apps, counts)
        self.lease_grant(active_apps, t, np.zeros(len(active_apps), dtype=np.int64))


class FixedTTL(BasePolicy):
    def __init__(self, n_apps, mem_mb, ttl=10, **kw):
        super().__init__(n_apps, mem_mb, **kw)
        self.ttl = ttl
        self.name = f"TTL-{ttl}m"

    def on_activity(self, t, active_apps, counts):
        super().on_activity(t, active_apps, counts)
        self.lease_grant(active_apps, t, np.full(len(active_apps), self.ttl, dtype=np.int64))


class HybridHistogram(BasePolicy):
    """Shahrad et al. (ATC'20) 混合直方图式保活策略的在线复现。

    当 P(gap <= w) >= theta 时授予 w 分钟租约，w 取满足条件的最小窗口。
    """
    name = "Historic-TTL (ATC'20-style)"

    def __init__(self, n_apps, mem_mb, theta=0.5, windows=(1, 5, 15, 30, 60), **kw):
        super().__init__(n_apps, mem_mb, **kw)
        self.theta = theta
        self.windows = np.array(windows, dtype=np.int64)

    def on_activity(self, t, active_apps, counts):
        super().on_activity(t, active_apps, counts)
        lengths = np.ones(len(active_apps), dtype=np.int64)
        remaining = np.ones(len(active_apps), dtype=bool)
        for w in self.windows:
            if not remaining.any():
                break
            p, tot = self.reuse_prob(active_apps[remaining], w)
            ok = (p >= self.theta) & (tot >= 3)
            idx = np.nonzero(remaining)[0][ok]
            lengths[idx] = w
            remaining[np.nonzero(remaining)[0][ok]] = False
        self.lease_grant(active_apps, t, lengths)


class BudgetedPolicy(BasePolicy):
    """内存预算受限策略的公共骨架。

    调用发生时按各策略自己的规则授予租约（admission）；
    每 delta 分钟做一次预算执行（enforcement）：
      1) 若已授租约的内存超预算，按 rank 得分从低到高撤销租约；
      2) 若仍有余量，按 rank 得分从高到低主动为候选应用延长/授予租约。
    子类只需要提供 rank_score() 与 lease_length()。
    """
    budget_limited = True

    def __init__(self, n_apps, mem_mb, budget_mb, delta=5, min_obs=3, **kw):
        super().__init__(n_apps, mem_mb, **kw)
        self.budget_mb = float(budget_mb)
        self.delta = delta
        self.min_obs = min_obs

    # --- 子类实现 ---
    def rank_score(self, apps):
        raise NotImplementedError

    def lease_length(self, apps):
        return np.full(len(apps), 1, dtype=np.int64)

    def candidate_apps(self, t):
        return np.nonzero(self.last_active >= t - 24 * 60)[0]

    # --- 通用实现 ---
    def on_activity(self, t, active_apps, counts):
        BasePolicy.on_activity(self, t, active_apps, counts)
        self.lease_grant(active_apps, t, self.lease_length(active_apps))

    def replan(self, t, active_apps, counts):
        self.enforce_budget(t)
        # 有余量则主动延长高分应用的租约（proactive lease extension）
        warm = np.nonzero(self.warm_until >= t)[0]
        used = self.mem_mb[warm].sum()
        budget_left = self.budget_mb - used
        if budget_left <= 0:
            return
        cand = np.setdiff1d(self.candidate_apps(t), warm, assume_unique=False)
        if len(cand) == 0:
            return
        score = self.rank_score(cand)
        order = np.argsort(-score)
        cum = np.cumsum(self.mem_mb[cand[order]])
        n_fit = int(np.searchsorted(cum, budget_left, side="right"))
        select = cand[order][:n_fit]
        select = select[score[order][:n_fit] > 0]
        if len(select):
            self.lease_grant(select, t, self.lease_length(select))

    def enforce_budget(self, t):
        """每分钟执行：超预算则撤销得分最低的租约。"""
        warm = np.nonzero(self.warm_until >= t)[0]
        if len(warm) == 0:
            return
        used = float(self.mem_mb[warm].sum())
        if used <= self.budget_mb:
            return
        score = self.rank_score(warm)
        order = np.argsort(score)                    # 低分先撤销
        cum = np.cumsum(self.mem_mb[warm[order]])
        excess = cum > self.budget_mb
        if excess.any():
            self.warm_until[warm[order][excess]] = t - 1


class TopKStatic(BudgetedPolicy):
    """按预测调用率（popularity）排序的预算内保活。"""
    name = "Popularity-ranked"

    def __init__(self, n_apps, mem_mb, budget_mb, delta=5, ttl=60, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, delta=delta, **kw)
        self.ttl = ttl

    def rank_score(self, apps):
        # 内存感知的热度基线：单位内存的预测调用率
        return self.ewma_rate[apps] / np.maximum(self.mem_mb[apps], 1.0)

    def lease_length(self, apps):
        return np.full(len(apps), self.ttl, dtype=np.int64)


class LRUProfit(BudgetedPolicy):
    """按最近活跃时间（LRU）排序的预算内保活。"""
    name = "LRU-ranked"

    def __init__(self, n_apps, mem_mb, budget_mb, delta=5, ttl=15, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, delta=delta, **kw)
        self.ttl = ttl

    def rank_score(self, apps):
        return self.last_active[apps].astype(np.float64)

    def lease_length(self, apps):
        return np.full(len(apps), self.ttl, dtype=np.int64)


class BALM(BudgetedPolicy):
    """BALM: Budget-Aware Lease Management（本文方法）。

    在每个重规划点，用"每分钟期望节省的时延 / 内存占用"排序，在内存预算内
    贪心分配租约；租约长度取该应用的复用时间窗（P(gap<=w) >= theta 的最小 w），
    使可预测性差的应用自动获得短租约，可预测性强的（如定时触发）获得长租约。
    """
    name = "BALM (ours)"

    def __init__(self, n_apps, mem_mb, budget_mb, delta=5, theta=0.8,
                 windows=(1, 2, 5, 15, 30, 60, 180, 720), slo_weighted=False,
                 min_obs=3, cold_penalty=None, slo=SLO_SEC, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, delta=delta, min_obs=min_obs, **kw)
        self.theta = theta
        self.windows = np.array(windows, dtype=np.int64)
        self.slo_weighted = slo_weighted
        self.cold_penalty = cold_penalty if cold_penalty is not None else np.ones(n_apps)
        self.slo = slo
        self.mean_s = np.zeros(n_apps, dtype=np.float64)

    def _lease_length(self, apps):
        """对每个应用求 P(gap<=w)>=theta 的最小窗口 w。"""
        lengths = np.ones(len(apps), dtype=np.int64)
        remaining = np.ones(len(apps), dtype=bool)
        for w in self.windows:
            if not remaining.any():
                break
            sub = apps[remaining]
            p, tot = self.reuse_prob(sub, w)
            ok = (p >= self.theta) & (tot >= self.min_obs)
            idx = np.nonzero(remaining)[0][ok]
            lengths[idx] = w
            remaining[np.nonzero(remaining)[0][ok]] = False
        return lengths

    def set_aux(self, mean_s, cold_penalty):
        self.mean_s = mean_s
        self.cold_penalty = cold_penalty
        self.window = 60
        self.act_ring = np.zeros((self.n_apps, self.window), dtype=np.uint8)

    def tick(self, t, active_apps):
        """维护 60 分钟滑动窗口的"活跃分钟"指示 -> P(应用在某一分钟活跃)。"""
        slot = t % self.window
        self.act_ring[:, slot] = 0
        self.act_ring[active_apps, slot] = 1

    def p_active(self, apps):
        return self.act_ring[apps].mean(axis=1)

    # 排序得分：单位内存的期望收益（每 MB 每分钟避免的冷启动时延）
    def rank_score(self, apps):
        # 保活 1 分钟避免的期望冷启动时延 = P(该分钟活跃) x E[冷启动时延]；
        # 成本 = 该应用常驻内存。二者相除即为收益密度。
        p_act = self.p_active(apps)
        cold = self.cold_penalty[apps]
        if self.slo_weighted:
            dur = self.mean_s[apps]
            # 冷启动是否把这次调用推过 SLO 的期望强度
            excess = np.clip((dur + cold - self.slo) / np.maximum(cold, 1e-6), 0.0, 1.0)
            return p_act * excess * cold / np.maximum(self.mem_mb[apps], 1.0)
        return p_act * cold / np.maximum(self.mem_mb[apps], 1.0)

    # 租约长度：该应用的复用时间窗（P(gap<=w) >= theta 的最小 w）
    def lease_length(self, apps):
        return self._lease_length(apps)


# --------------------------------------------------------------------------
# 仿真主循环
# --------------------------------------------------------------------------
def _bin_axis(nb):
    return (np.arange(nb) + 0.5) / nb


def _brier_from_bins(pos, neg, nb):
    """由 1000 分箱近似计算 Brier 分数（分箱中点作为预测概率）。"""
    n = pos.sum() + neg.sum()
    if n <= 0:
        return float("nan")
    p = _bin_axis(nb)
    return float(np.sum(neg * p ** 2 + pos * (1.0 - p) ** 2) / n)


def _auroc_from_bins(pos, neg):
    """由分箱直方图计算 AUROC（梯形积分）。"""
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    order = np.argsort(-_bin_axis(len(pos)))          # 分数从高到低
    tp = np.cumsum(pos[order]) / P
    fp = np.cumsum(neg[order]) / N
    return float(np.trapezoid(np.r_[0.0, tp], np.r_[0.0, fp]))


def _auprc_from_bins(pos, neg):
    """由分箱直方图计算 average precision（阶梯法）。"""
    P = pos.sum()
    if P <= 0:
        return float("nan")
    order = np.argsort(-_bin_axis(len(pos)))
    tp = np.cumsum(pos[order])
    fp = np.cumsum(neg[order])
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / P
    ap = 0.0
    prev_r = 0.0
    for pr, rc in zip(prec, rec):
        ap += pr * (rc - prev_r)
        prev_r = rc
    return float(ap)


def simulate(trace, policy, slo=SLO_SEC, seed=0, verbose=False, ndays=None,
             cold_median_s=COLD_MEDIAN_S, cold_penalty_mode="memory",
             init_s=1.0, eval_start=0):
    """分钟级仿真。

    eval_start: 预热窗口长度（分钟）。所有策略的状态统计从 t=0 开始更新，
    但指标只在 t >= eval_start 的分钟上累计。用于"训练/评测分离"的协议，
    例如 eval_start=5*1440 表示：前 5 天仅用于建立统计/训练模型，指标在
    第 6-14 天统计。
    """
    counts = trace["counts"]
    mem = trace["mem_mb"]
    q_s = trace["q_s"]
    q_w = trace["q_w"].astype(np.float64)
    q_w = np.diff(np.concatenate([[0.0], q_w]))   # 每个分位点的权重
    n_min = trace["n_minutes"]
    n_apps = trace["n_apps"]

    rng = np.random.default_rng(seed)
    cold_pen = draw_cold_penalties(mem, rng, median_s=cold_median_s,
                                   mode=cold_penalty_mode)
    if hasattr(policy, "set_aux"):
        policy.set_aux(trace["mean_s"], cold_pen)

    # 时延直方图（对数分箱 1ms .. 60s），热路径与冷启动路径分别累计
    nbins = 220
    edges = np.logspace(np.log10(0.001), np.log10(60.0), nbins + 1)
    hist = np.zeros(nbins, dtype=np.float64)
    nq = q_s.shape[1]
    bin_idx = [np.clip(np.digitize(q_s[:, j], edges) - 1, 0, nbins - 1) for j in range(nq)]
    cold_bin = [np.clip(np.digitize(q_s[:, j] + cold_pen, edges) - 1, 0, nbins - 1)
                for j in range(nq)]

    indptr = counts.indptr
    indices = counts.indices
    data = counts.data
    EMPTY_APPS = np.empty(0, dtype=np.int32)

    warm_until = policy.warm_until
    cold_events = 0
    cold_slo = 0.0
    cold_app_seen = np.zeros(n_apps, dtype=bool)
    cold_per_app = np.zeros(n_apps, dtype=np.int64)
    # 容器生命周期记账：retention（常驻保留） vs prewarm（主动预启动）
    resident_prev = np.zeros(n_apps, dtype=bool)
    prewarm_attempts = 0
    wasted_prewarms = 0
    retained = 0
    subminute_extra = 0.0     # 亚分钟到达下，因容器尚未就绪而额外产生的冷启动
    util_samples = []
    cand_sizes = []
    budget_mb = getattr(policy, "budget_mb", None)
    active_flag = np.zeros(n_apps, dtype=bool)
    # 概率审计：1000 个分数分箱内累计正/负样本数（用于 AUROC/AUPRC）
    NB = 1000
    prob_pos = np.zeros(NB, dtype=np.float64)
    prob_neg = np.zeros(NB, dtype=np.float64)
    total_inv = 0.0
    latency_ms_sum = 0.0        # Σ 调用数 x 执行耗时(ms)
    cold_ms_sum = 0.0           # Σ 冷启动额外耗时
    mem_minutes = np.zeros(n_min, dtype=np.float64)
    exec_minutes = np.zeros(n_min, dtype=np.float64)
    replan_every = getattr(policy, "delta", 0)
    active_count = 0

    for t in range(n_min):
        counting = t >= eval_start
        s, e = indptr[t], indptr[t + 1]
        # 注意：即使本分钟没有调用，也必须执行决策与记账——否则规划器的“每分钟重新决策”、
        # 生存计数递减和常驻内存记账都会被跳过（对稀疏 trace 会造成静默错误）。
        if s == e:
            apps = EMPTY_APPS
            cnt = np.empty(0, dtype=np.float64)
        else:
            apps = indices[s:e]
            cnt = data[s:e].astype(np.float64)

        if replan_every and (t % replan_every == 0):
            policy.replan(t, apps, cnt)
            k = getattr(policy, "last_cand_size", None)
            if k:
                cand_sizes.append(k)

        # --- 容器生命周期：新常驻（=主动预热）与应用被保留 ---
        active_flag[:] = False
        active_flag[apps] = True
        warm_mask = warm_until >= t
        newly = warm_mask & ~resident_prev
        n_new = int(newly.sum())
        if n_new:
            if counting:
                prewarm_attempts += n_new
                wasted_prewarms += n_new - int(active_flag[newly].sum())
            # 亚分钟到达效应：预热决策在分钟 t 开始时生效，容器需要 init_s 秒就绪；
            # 若调用在就绪前到达（均匀到达假设下比例为 init_s/60），仍会命中冷容器。
            pre_apps = np.nonzero(newly)[0]
            in_pre = np.isin(apps, pre_apps)
            if counting and in_pre.any():
                p_early = float(np.clip(init_s / 60.0, 0.0, 1.0))
                c_pre = cnt[in_pre]
                subminute_extra += float(np.sum(1.0 - (1.0 - p_early) ** c_pre))
        if counting:
            retained += int((warm_mask & resident_prev).sum())

        is_warm = warm_until[apps] >= t
        cold_mask = ~is_warm
        n_cold = int(cold_mask.sum())
        if n_cold:
            if counting:
                cold_events += n_cold
                cold_apps = apps[cold_mask]
                cold_app_seen[cold_apps] = True
                np.add.at(cold_per_app, cold_apps, 1)
                cold_ms_sum += float(cold_pen[cold_apps].sum())
                cold_slo += float(((trace["mean_s"][cold_apps] + cold_pen[cold_apps]) > slo).sum())
            else:
                cold_apps = apps[cold_mask]
        else:
            cold_apps = None

        # 每个应用-分钟的调用数按该应用的耗时分位分布摊到直方图；
        # 冷启动的那一次调用从"耗时分布"移到"耗时+冷启动"分布。
        for j in range(nq):
            if counting:
                hist += np.bincount(bin_idx[j][apps], weights=cnt * q_w[j], minlength=nbins)
                if n_cold:
                    hist += np.bincount(cold_bin[j][cold_apps], weights=np.full(n_cold, q_w[j]),
                                        minlength=nbins)
                    hist -= np.bincount(bin_idx[j][cold_apps], weights=np.full(n_cold, q_w[j]),
                                        minlength=nbins)

        if counting:
            latency_ms_sum += float((cnt * trace["mean_ms"][apps]).sum())
            total_inv += float(cnt.sum())
            active_count += len(apps)
        # 常驻内存 = 保活容器（受预算约束的部分）
        mem_warm = float(mem[warm_mask].sum())
        if counting:
            mem_minutes[t] = mem_warm
            if budget_mb:
                util_samples.append(mem_warm / budget_mb)
        # 执行内存 = 本分钟为执行而新启动的冷容器
        if counting and n_cold:
            exec_minutes[t] += float(mem[cold_apps].sum())

        if counting and getattr(policy, "_last_pred", None) is not None:
            pr = np.asarray(policy._last_pred, dtype=np.float64)
            cd = np.asarray(policy._last_cand)
            hit = np.isin(cd, apps)
            b = np.clip((pr * NB).astype(np.int64), 0, NB - 1)
            np.add.at(prob_pos, b[hit], 1.0)
            np.add.at(prob_neg, b[~hit], 1.0)
        # 状态更新放在“分钟 t 的决策之后”：tick 会把分钟 t 的活跃写进滑动窗口，
        # 而 t 时刻的特征/决策只能使用 t-1 及更早的数据（否则学习式基线的
        # “昨天同一分钟”特征会读到当前分钟，形成标签泄漏）。
        policy.tick(t, apps)
        policy.on_activity(t, apps, cnt)
        if hasattr(policy, "observe_outcome"):
            if counting:
                policy.observe_outcome(t, apps)
        resident_prev[:] = warm_until >= t
        if policy.budget_limited:
            policy.enforce_budget(t + 1)

    # 结果汇总
    total_weight = hist.sum()
    cum = np.cumsum(hist) / max(total_weight, 1.0)
    def qtl(p):
        i = int(np.searchsorted(cum, p))
        return float(edges[min(i, nbins - 1)])

    mean_lat = (latency_ms_sum + cold_ms_sum) / max(total_inv, 1e-9) / 1000.0
    over = float(hist[(edges[:-1] >= slo)].sum())
    p999 = qtl(0.999)
    k_arr = np.array(cand_sizes, dtype=np.float64) if cand_sizes else np.array([0.0])
    eval_minutes = max(n_min - eval_start, 1)
    # 内存指标只在评测窗口内统计（预热窗口不计入均值/分位数）
    mem_eval = mem_minutes[eval_start:]
    exec_eval = exec_minutes[eval_start:]
    lp_gap = getattr(policy, "lp_gap_sum", None)
    cal_count = getattr(policy, "cal_count", None)
    cal_hit = getattr(policy, "cal_hit", None)
    res = dict(
        policy=policy.name,
        cold_events=cold_events,
        total_invocations=total_inv,
        active_app_minutes=active_count,
        cold_ratio_per_1k=cold_events / max(total_inv, 1e-9) * 1000.0,
        cold_minute_ratio=cold_events / max(active_count, 1e-9),
        mean_latency_s=mean_lat,
        p50_latency_s=qtl(0.50),
        p95_latency_s=qtl(0.95),
        p99_latency_s=qtl(0.99),
        p999_latency_s=p999,
        slo_violation_rate=over / max(total_weight, 1e-9),
        mean_prewarm_mem_gb=float(mem_eval.mean()) / 1024.0,
        p95_prewarm_mem_gb=float(np.percentile(mem_eval, 95)) / 1024.0,
        max_prewarm_mem_gb=float(mem_eval.max()) / 1024.0,
        gb_hours=float(mem_eval.sum()) / 1024.0 / 60.0,
        mean_exec_mem_gb=float(exec_eval.mean()) / 1024.0,
        mean_total_mem_gb=float(mem_eval.mean() + exec_eval.mean()) / 1024.0,
        exec_gb_hours=float(exec_eval.sum()) / 1024.0 / 60.0,
        apps_with_cold=int(cold_app_seen.sum()),
        apps_with_cold_share=float(cold_app_seen.mean()),
        cold_per_app_p50=float(np.percentile(cold_per_app, 50)),
        cold_per_app_p90=float(np.percentile(cold_per_app, 90)),
        cold_per_app_p99=float(np.percentile(cold_per_app, 99)),
        cold_per_app_max=float(cold_per_app.max()),
        cold_slo_events=cold_slo,
        cold_slo_per_1k=cold_slo / max(total_inv, 1e-9) * 1000.0,
        mean_cold_added_ms=cold_ms_sum / max(total_inv, 1e-9),
        cold_per_app_day=cold_events / max(n_apps, 1) / (eval_minutes / 1440.0),
        eval_start=eval_start,
        eval_minutes=eval_minutes,
        # 容器生命周期与预热开销
        prewarm_attempts=prewarm_attempts,
        prewarm_per_1k=prewarm_attempts / max(total_inv, 1e-9) * 1000.0,
        wasted_prewarms=wasted_prewarms,
        wasted_prewarm_rate=wasted_prewarms / max(prewarm_attempts, 1),
        retained_container_minutes=retained,
        total_inits=cold_events + prewarm_attempts,
        total_inits_per_1k=(cold_events + prewarm_attempts) / max(total_inv, 1e-9) * 1000.0,
        budget_utilization=float(np.mean(util_samples)) if util_samples else float("nan"),
        cand_k_mean=float(k_arr.mean()),
        cand_k_p95=float(np.percentile(k_arr, 95)),
        cand_k_max=float(k_arr.max()),
        knapsack_gap_mean=(lp_gap[0] / lp_gap[1]) if (lp_gap and lp_gap[1] > 0) else float("nan"),
        brier_score=(policy.brier_sum / policy.brier_n) if getattr(policy, "brier_n", 0) else float("nan"),
        cal_pred_frac=(float(cal_hit.sum() / cal_count.sum())) if (cal_count is not None
                                                                  and cal_count.sum() > 0) else float("nan"),
        audited_candidates=float(prob_pos.sum() + prob_neg.sum()),
        predictor_brier=_brier_from_bins(prob_pos, prob_neg, NB),
        predictor_auroc=_auroc_from_bins(prob_pos, prob_neg),
        predictor_auprc=_auprc_from_bins(prob_pos, prob_neg),
        knapsack_gap_max=float(getattr(policy, "lp_gap_max", float("nan"))),
        subminute_extra_cold=subminute_extra,
        subminute_extra_per_1k=subminute_extra / max(total_inv, 1e-9) * 1000.0,
        cold_plus_subminute_per_1k=(cold_events + subminute_extra) / max(total_inv, 1e-9) * 1000.0,
    )
    if cal_count is not None and cal_count.sum() > 0:
        for i in range(len(cal_count)):
            res[f"cal_bin{i}_count"] = int(cal_count[i])
            res[f"cal_bin{i}_hit"] = int(cal_hit[i])
    return res
