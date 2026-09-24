#!/usr/bin/env python3
"""
policies_plan.py — 逐分钟规划型（planner）策略。

与 sim.py 中的租约型策略不同，planner 每分钟重新决定"这一分钟把哪些应用
保持热"，在内存预算内做背包近似（按收益密度贪心）。逐分钟规划能覆盖
"应用尚未来访、但可以预测其即将活跃"的情况，这是 TTL / LRU 做不到的。

  OracleKnapsack : 使用真实活跃集合（非因果，作为可达上限参照）
  BALMPlanner    : 使用因果预测 P(应用在该分钟活跃) 做规划
"""

from __future__ import annotations

import numpy as np

from sim import BudgetedPolicy, BasePolicy, MIN_DUR_S


def greedy_fill(mem, value, budget, modified=True):
    """在内存预算内按 value/mem 密度贪心装箱。

    modified=True 时额外与"单个最优项"比较并取较优解，据此可获得 0/1 背包
    问题的 1/2 近似保证（仅密度贪心本身没有该保证）。同时返回分数背包
    （LP 松弛）目标值，作为整数最优解的上界，用于实测最优性差距。
    返回 (selected_index, chosen_value, lp_bound)。
    """
    n = len(mem)
    if n == 0:
        return np.empty(0, dtype=np.int64), 0.0, 0.0
    mem = np.asarray(mem, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    density = value / np.maximum(mem, 1.0)
    order = np.argsort(-density)
    mem_o, val_o = mem[order], value[order]
    cum = np.cumsum(mem_o)
    k = int(np.searchsorted(cum, budget, side="right"))
    sel = order[:k]
    chosen = float(val_o[:k].sum())
    if k < n:
        rem = budget - (cum[k - 1] if k > 0 else 0.0)
        lp = chosen + float(val_o[k]) * min(1.0, max(0.0, rem / max(mem_o[k], 1.0)))
    else:
        lp = chosen
    if modified:
        best = int(np.argmax(value))
        if value[best] > chosen:
            sel = np.array([best], dtype=np.int64)
            chosen = float(value[best])
    return sel, chosen, float(max(lp, chosen))


class PlannerBase(BudgetedPolicy):
    """逐分钟规划：replan() 直接决定当前分钟的热集合。"""
    delta = 1

    def __init__(self, n_apps, mem_mb, budget_mb, cold_penalty=None, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, delta=1, **kw)
        self.cold_penalty = (cold_penalty if cold_penalty is not None
                             else np.ones(n_apps, dtype=np.float32))

    def set_aux(self, mean_s, cold_penalty):
        self.cold_penalty = cold_penalty
        self.mean_s = mean_s

    def on_activity(self, t, active_apps, counts):
        BasePolicy.on_activity(self, t, active_apps, counts)
        # planner 不在此授予租约；热集合完全由 replan() 决定

    def _select(self, cand, score, t):
        """按策略自身的排序分数 score 贪心装入预算（baseline 保持原始语义）。"""
        self.last_cand_size = len(cand)
        if len(cand) == 0:
            return np.empty(0, dtype=np.int64)
        order = np.argsort(-score)
        cum = np.cumsum(self.mem_mb[cand[order]])
        n = int(np.searchsorted(cum, self.budget_mb, side="right"))
        sel = cand[order][:n]
        return sel[score[order][:n] > 0]

    def _select_value(self, cand, value, t):
        """按 value/mem 密度做修正贪心装箱（1/2 保证），并累计与 LP 上界的差距。

        只用于目标函数可解释为"期望避免时延"的策略（BALM 与 oracle）。
        """
        self.last_cand_size = len(cand)
        if len(cand) == 0:
            return np.empty(0, dtype=np.int64)
        sel, chosen, lp = greedy_fill(self.mem_mb[cand], value, self.budget_mb,
                                      modified=True)
        if not hasattr(self, "lp_gap_sum"):
            self.lp_gap_sum = [0.0, 0.0]
            self.lp_gap_max = 0.0
        gap = (lp - chosen) / lp if lp > 0 else 0.0
        self.lp_gap_sum[0] += (lp - chosen)
        self.lp_gap_sum[1] += lp
        self.lp_gap_max = max(self.lp_gap_max, gap)
        return cand[sel]

    def _apply(self, t, sel):
        # 只保活当前分钟（下一分钟重新决策）
        self.warm_until[:] = t - 1
        if len(sel):
            self.warm_until[sel] = t


class OracleKnapsack(PlannerBase):
    """非因果上限参照：已知这一分钟真实活跃的应用，按 T_cold/MB 装箱。"""
    name = "Oracle knapsack"

    def replan(self, t, active_apps, counts):
        value = self.cold_penalty[active_apps].astype(np.float64)   # 已知活跃：值为 d_a
        sel = self._select_value(active_apps, value, t)
        self._apply(t, sel)


class BALMPlanner(PlannerBase):
    """BALM：预算感知的预测式热集合规划。

    每分钟：
      1) 预测每个候选应用在"这一分钟"的活跃概率 p_a
         p_a = 1 - (1-p_season)(1-p_momentum)
         · p_season  : 一天内分钟级周期画像（在线累计，跨天衰减）
         · p_momentum: 近期活跃的衰减动量（近邻效应）
      2) 期望收益 = p_a x E[冷启动时延]，成本 = 常驻内存，
         按 收益/内存 贪心装箱填满预算。
    与 LRU/TTL 的本质区别：LRU 只能对"已经活跃过"的应用保活，
    BALM 可以依据周期画像在应用到来之前就把它准备好。
    """
    name = "BALM (ours)"

    def __init__(self, n_apps, mem_mb, budget_mb, cold_penalty=None,
                 tau_min=2.0, prior_w=1.0, season_thresh=1e-3,
                 use_season=True, use_momentum=True, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, cold_penalty=cold_penalty, **kw)
        self.tau = float(tau_min)
        self.prior_w = float(prior_w)
        self.season_thresh = float(season_thresh)
        self.use_season = use_season
        self.use_momentum = use_momentum
        # 分钟级周期画像：过去若干天在同一 minute-of-day 出现的次数
        self.season = np.zeros((n_apps, 1440), dtype=np.uint16)
        self.glob_season = np.zeros(1440, dtype=np.float64)
        self.days_obs = 0.0
        self.last_decay_day = -1
        self.minutes_seen = 0

    # ---- 预测 ----
    def tick(self, t, active_apps):
        mod = t % 1440
        day = t // 1440
        self.minutes_seen += 1
        if day != self.last_decay_day:
            # 跨天：历史计数衰减，保持对近期行为的敏感
            if self.last_decay_day >= 0:
                self.season >>= 1
                self.glob_season *= 0.5
            self.days_obs += 1.0
            self.last_decay_day = day
        if len(active_apps):
            np.add.at(self.season, (active_apps, mod), np.uint16(1))
        self.glob_season[mod] += len(active_apps)

    def _p_season(self, apps, mod):
        cnt = self.season[apps, mod].astype(np.float64)
        prior = self.glob_season[mod] / max(self.days_obs, 1.0)
        denom = max(self.days_obs, 1.0) + self.prior_w
        # 拉普拉斯式平滑：样本少时向"该 minute-of-day 的全局活跃强度"收缩
        return (cnt + self.prior_w * min(prior, 1.0)) / denom

    def _p_momentum(self, apps, t):
        last = self.last_active[apps]
        age = np.clip(t - last, 0, None)
        return np.where(last < 0, 0.0, np.exp(-age / max(self.tau, 1e-6)))

    def replan(self, t, active_apps, counts):
        mod = t % 1440
        # 候选：近期活跃过的 + 周期画像中该分钟出现过的
        recent = np.nonzero(self.last_active >= t - 120)[0] if self.use_momentum \
            else np.empty(0, dtype=np.int64)
        seasonal = np.nonzero(self.season[:, mod] > 0)[0] if self.use_season \
            else np.empty(0, dtype=np.int64)
        cand = np.union1d(recent, seasonal)
        if len(cand) == 0:
            self._apply(t, np.empty(0, dtype=np.int64))
            return
        p_season = self._p_season(cand, mod) if self.use_season else np.zeros(len(cand))
        p_mom = self._p_momentum(cand, t) if self.use_momentum else np.zeros(len(cand))
        p = 1.0 - (1.0 - p_season) * (1.0 - p_mom)
        p[p < self.season_thresh] = 0.0
        score = p * self.cold_penalty[cand] / np.maximum(self.mem_mb[cand], 1.0)
        sel = self._select(cand, score, t)
        self._apply(t, sel)


class PlannerPopularity(PlannerBase):
    """同一逐分钟装箱机制，但排序信号换成"内存感知的热度"。"""
    name = "Planner-Popularity"

    def __init__(self, n_apps, mem_mb, budget_mb, cold_penalty=None,
                 recall_min=120, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, cold_penalty=cold_penalty, **kw)
        self.recall_min = recall_min

    def rank_score(self, apps):
        return self.ewma_rate[apps] / np.maximum(self.mem_mb[apps], 1.0)

    def replan(self, t, active_apps, counts):
        cand = np.nonzero(self.last_active >= t - self.recall_min)[0]
        self.last_cand_size = len(cand)
        if len(cand) == 0:
            self._apply(t, np.empty(0, dtype=np.int64))
            return
        score = self.rank_score(cand)
        self._apply(t, self._select(cand, score, t))


class BALMJIT(PlannerBase):
    """BALM-JIT：基于"到达间隔分布"的准时预热 (just-in-time warm-up)。

    与 TTL / LRU 的根本区别：
      · TTL/LRU 只能对"已经活跃过"的应用保活，容器在空闲间隙里一直占内存；
      · BALM-JIT 用每个应用自己的间隔分布预测它**下一分钟**会不会被调用，
        只在需要的那一分钟把容器准备好，内存按预测的收益密度在预算内分配。

    预测器（严格因果）：
      设 lag = t - last_active[a]，应用 a 在历史间隔中"恰好 lag 分钟"出现的
      频率即为 P(该分钟激活)。这正是 renewal 过程的一步转移概率估计。
      对没有历史的应用用全局先验；历史样本少时向全局分布收缩。

    打分：score = p x E[冷启动时延] / 内存占用
    """
    name = "BALM-JIT (ours)"
    MAX_GAP = 120          # 精确记录的间隔上限（分钟）

    def __init__(self, n_apps, mem_mb, budget_mb, cold_penalty=None,
                 prior_w=2.0, use_global_prior=True, per_app=True,
                 mem_aware=True, max_gap=None, use_hazard=True, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, cold_penalty=cold_penalty, **kw)
        if max_gap is not None:
            self.MAX_GAP = int(max_gap)
        self.gap_count = np.zeros((n_apps, self.MAX_GAP + 1), dtype=np.int16)
        self.gap_tail = np.zeros(n_apps, dtype=np.int32)      # >MAX_GAP 的次数
        self.gap_total = np.zeros(n_apps, dtype=np.int32)
        self.glob_gap = np.zeros(self.MAX_GAP + 1, dtype=np.float64)
        self.glob_total = 0.0
        self.prior_w = prior_w
        self.use_global_prior = use_global_prior
        self.per_app = per_app
        self.mem_aware = mem_aware
        self.use_hazard = use_hazard
        # 生存计数 S_a(lag) = #{gaps >= lag}，每分钟增量维护（避免每轮 cumsum）
        self.surv = np.zeros(n_apps, dtype=np.int32)
        # 预测校准统计：10 个概率分箱内的预测数、实际命中数、Brier 累加
        self.cal_bins = 10
        self.cal_count = np.zeros(self.cal_bins, dtype=np.int64)
        self.cal_hit = np.zeros(self.cal_bins, dtype=np.int64)
        self.brier_sum = 0.0
        self.brier_n = 0
        self._last_pred = None
        self._last_cand = None
        self.last_cand_size = 0
        self.n_pred = 0
        self.n_hit = 0

    # ---- 在线更新间隔分布 ----
    def on_activity(self, t, active_apps, counts):
        last = self.last_active[active_apps]
        gap = t - last
        known = gap > 0
        if known.any():
            apps_k = active_apps[known]
            gaps_k = gap[known]
            small = gaps_k <= self.MAX_GAP
            if small.any():
                np.add.at(self.gap_count, (apps_k[small], gaps_k[small]), np.int16(1))
            if (~small).any():
                np.add.at(self.gap_tail, apps_k[~small], 1)
            np.add.at(self.gap_total, apps_k, 1)
            self.glob_gap += np.bincount(np.clip(gaps_k, 0, self.MAX_GAP),
                                         minlength=self.MAX_GAP + 1)
            self.glob_total += len(gaps_k)
            # 刚活跃过的应用：lag 将从 1 开始，S = 全部观测到的间隔
            self.surv[apps_k] = self.gap_total[apps_k]
        # 记录预测命中（用于论文中的预测精度指标）
        n = len(active_apps)
        self.n_pred += n
        BasePolicy.on_activity(self, t, active_apps, counts)

    def _p_active(self, apps, t):
        lag = t - self.last_active[apps]
        lag = np.clip(lag, 0, self.MAX_GAP)
        prior_pmf = self.glob_gap[lag] / max(self.glob_total, 1.0) if self.use_global_prior \
            else np.full(len(apps), 1.0 / max(self.MAX_GAP, 1))
        # 全局生存函数 P(G >= lag)：LP 松弛需要的先验分母
        if self.use_global_prior:
            tail_glob = self.glob_gap[::-1]
            surv_glob = np.cumsum(tail_glob)[::-1] / max(self.glob_total, 1.0)
            prior_surv = surv_glob[lag]
        else:
            prior_surv = np.clip((self.MAX_GAP - lag + 1) / max(self.MAX_GAP, 1), 1e-9, None)
        if not self.per_app:
            return prior_pmf / np.maximum(prior_surv, 1e-9)     # 纯全局 hazard

        num = self.gap_count[apps, lag].astype(np.float64)
        tot = self.gap_total[apps].astype(np.float64)
        surv = self.surv[apps].astype(np.float64)
        if self.use_hazard:
            # 条件风险 P(G = lag | G >= lag)：控制器已观察到前 lag-1 分钟没有调用
            p = (num + self.prior_w * prior_pmf) / (surv + self.prior_w * prior_surv)
        else:
            # 边缘 PMF P(G = lag)（早期实现，保留用于消融对比）
            p = (num + self.prior_w * prior_pmf) / (tot + self.prior_w)
        p = np.clip(p, 0.0, 1.0)
        # 首次出现（无历史）的应用：用全局 hazard
        no_hist = self.last_active[apps] < 0
        if no_hist.any():
            p = np.where(no_hist, prior_pmf / np.maximum(prior_surv, 1e-9), p)
        return p

    def replan(self, t, active_apps, counts):
        lag_all = t - self.last_active
        # lag >= 1：分钟 t 的决策发生在 t-1 的统计更新之后，因此不会出现 lag = 0
        track = (lag_all >= 1) & (lag_all <= self.MAX_GAP + 1)
        if track.any():
            idx = np.clip(lag_all[track] - 1, 0, self.MAX_GAP)
            self.surv[track] = self.surv[track] - self.gap_count[track, idx]
        cand = np.nonzero((lag_all >= 1) & (lag_all <= self.MAX_GAP))[0]
        self.last_cand_size = len(cand)
        if len(cand) == 0:
            self._apply(t, np.empty(0, dtype=np.int64))
            return
        p = self._p_active(cand, t)
        self._last_pred = p
        self._last_cand = cand
        value = p * self.cold_penalty[cand]
        if self.mem_aware:
            self._apply(t, self._select_value(cand, value, t))
        else:
            # 关闭内存感知：按绝对期望收益排序（相当于把内存视为相同）
            order = np.argsort(-value)
            cum = np.cumsum(self.mem_mb[cand[order]])
            n = int(np.searchsorted(cum, self.budget_mb, side="right"))
            sel = cand[order][:n]
            self._apply(t, sel[value[order][:n] > 0])

    def observe_outcome(self, t, active_apps):
        """分钟 t 结束后回填预测校准（预测在决策时已冻结，故无信息泄漏）。"""
        if self._last_pred is None or self._last_cand is None:
            return
        pred = self._last_pred
        cand = self._last_cand
        hit = np.zeros(len(cand), dtype=np.float64)
        hit[np.isin(cand, active_apps, assume_unique=False)] = 1.0
        bins = np.clip((pred * self.cal_bins).astype(np.int64), 0, self.cal_bins - 1)
        np.add.at(self.cal_count, bins, 1)
        np.add.at(self.cal_hit, bins, hit.astype(np.int64))
        self.brier_sum += float(np.sum((pred - hit) ** 2))
        self.brier_n += len(cand)
        self._last_pred = None
        self._last_cand = None


class PlannerLRU(PlannerBase):
    """同一逐分钟装箱机制，排序信号换成"最近活跃时间"（经典 LRU）。"""
    name = "Planner-LRU"

    def __init__(self, n_apps, mem_mb, budget_mb, cold_penalty=None,
                 recall_min=120, mem_aware=False, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, cold_penalty=cold_penalty, **kw)
        self.recall_min = recall_min
        self.mem_aware = mem_aware

    def rank_score(self, apps):
        s = self.last_active[apps].astype(np.float64)
        if self.mem_aware:
            s = s / np.maximum(self.mem_mb[apps], 1.0)
        return s

    def replan(self, t, active_apps, counts):
        cand = np.nonzero(self.last_active >= t - self.recall_min)[0]
        self.last_cand_size = len(cand)
        if len(cand) == 0:
            self._apply(t, np.empty(0, dtype=np.int64))
            return
        score = self.rank_score(cand)
        self._apply(t, self._select(cand, score, t))
