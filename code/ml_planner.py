#!/usr/bin/env python3
"""
ml_planner.py — 学习式预测基线（审稿意见：需要一个更强的 predictive prewarming baseline）。

做法（严格因果）:
  1) 用 trace 前 5 天做训练窗口，按分钟抽样生成 (app, minute) 样本：
     正样本 = 该应用在该分钟活跃；负样本 = 候选池中随机抽到的非活跃应用-分钟。
  2) 特征只用该应用在 t-1 及更早的信息：lag、近 1/5/15/60 分钟活跃度、
     昨天同一分钟是否活跃、一天中的分钟（sin/cos）、星期几、历史平均间隔、
     平均执行时长、内存占用。
  3) 训练逻辑回归（在线可用、可解释、代价低），之后冻结模型用于第 6-14 天，
     与 BALM 使用完全相同的逐分钟预算装箱机制，只替换预测器。

输出: 与 BALM 可直接比较的 PlannerML 策略。
"""
from __future__ import annotations

import numpy as np

from policies_plan import PlannerBase
from sim import BasePolicy


def _feat_names():
    return ["lag", "act1", "act5", "act15", "act60", "same_min_yday",
            "mod_sin", "mod_cos", "dow", "mean_gap", "mean_dur", "mem"]


class FeatureTracker:
    """在线特征：只用 t-1 及更早的信息。"""

    def __init__(self, n_apps, mem_mb, mean_s, hist_len=1440):
        self.n_apps = n_apps
        self.mem = np.log1p(mem_mb.astype(np.float64))
        self.mean_dur = np.log1p(mean_s.astype(np.float64) * 1000.0)
        self.last_active = np.full(n_apps, -10**9, dtype=np.int64)
        self.gap_sum = np.zeros(n_apps, dtype=np.float64)
        self.gap_n = np.zeros(n_apps, dtype=np.int64)
        self.ring = np.zeros((n_apps, hist_len), dtype=np.uint8)   # 近 24h 活跃指示
        self.L = hist_len

    def tick(self, t, active_apps):
        slot = t % self.L
        self.ring[:, slot] = 0
        if len(active_apps):
            self.ring[active_apps, slot] = 1

    def update_activity(self, t, active_apps):
        prev = self.last_active[active_apps]
        gap = t - prev
        known = gap > 0
        if known.any():
            self.gap_sum[active_apps[known]] += gap[known]
            self.gap_n[active_apps[known]] += 1
        self.last_active[active_apps] = t

    def features(self, apps, t):
        """返回 (len(apps), 12) 的特征矩阵；窗口统计均只包含 t 之前的分钟。"""
        k = len(apps)
        f = np.empty((k, 12), dtype=np.float64)
        lag = np.clip(t - self.last_active[apps], 0, 10**6)
        f[:, 0] = np.log1p(lag)
        for j, W in enumerate((1, 5, 15, 60)):
            idx = (np.arange(t - W, t) % self.L)
            f[:, 1 + j] = self.ring[apps][:, idx].mean(axis=1)
        # 昨天同一分钟是否活跃（ring 长度 1440，故 (t-1440) % 1440 == t % 1440）
        same = self.ring[apps, t % self.L].astype(np.float64)
        f[:, 5] = same
        mod = t % 1440
        f[:, 6] = np.sin(2 * np.pi * mod / 1440.0)
        f[:, 7] = np.cos(2 * np.pi * mod / 1440.0)
        f[:, 8] = (t // 1440) % 7
        f[:, 9] = np.log1p(np.divide(self.gap_sum[apps], np.maximum(self.gap_n[apps], 1),
                                     out=np.zeros(k), where=self.gap_n[apps] > 0))
        f[:, 10] = self.mean_dur[apps]
        f[:, 11] = self.mem[apps]
        return f


def collect_training_data(trace, train_minutes, sample_stride=30, neg_per_min=400,
                          seed=0, offset=0, candidate_population=True, max_lag=120,
                          n_sample_minutes=240, minute_sampling="random",
                          verbose=False):
    """在训练窗口内按分钟抽样生成训练集（因果特征）。

    candidate_population=True 时，每个抽样分钟取**推理时的同一候选集**
    （1 <= lag <= max_lag 的全部应用，标签为当分钟是否活跃），从而让训练分布
    与推理分布一致、概率天然校准；此时 neg_per_min 被忽略。
    """
    counts = trace["counts"].tocsc()
    n_apps, n_min = counts.shape
    ft = FeatureTracker(n_apps, trace["mem_mb"], trace["mean_s"])
    rng = np.random.default_rng(seed)
    X, y = [], []
    indptr, indices = counts.indptr, counts.indices
    # 抽样分钟：默认随机抽样（固定种子）。固定步长会与 workload 的周期性（3/5/30/60 分钟
    # 定时任务）发生混叠，例如 t % 30 == 0 的分钟正例率高达 67%，而真实候选集只有 37%。
    if minute_sampling == "random":
        sampled = set(rng.choice(np.arange(1, min(train_minutes, n_min)),
                                 size=min(n_sample_minutes, max(train_minutes - 1, 1)),
                                 replace=False).tolist())
    else:
        sampled = set(range(offset, min(train_minutes, n_min), sample_stride))
    n_pos = n_rows = 0
    for t in range(min(train_minutes, n_min)):
        s, e = indptr[t], indptr[t + 1]
        act = indices[s:e]
        if t in sampled:
            if candidate_population:
                lag_all = t - ft.last_active
                cand = np.nonzero((lag_all >= 1) & (lag_all <= max_lag))[0]
            else:
                pool = np.nonzero(ft.last_active >= t - 1440)[0]
                if len(pool) > neg_per_min:
                    pool = rng.choice(pool, size=neg_per_min, replace=False)
                cand = np.union1d(act, pool)
            if len(cand):
                X.append(ft.features(cand, t))
                lab = np.isin(cand, act).astype(np.float64)
                y.append(lab)
                n_rows += len(cand)
                n_pos += int(lab.sum())
        ft.tick(t, act)
        ft.update_activity(t, act)
    if verbose:
        print(f"  训练抽样: {len(sampled)} 分钟, rows={n_rows:,}, positives={n_pos:,}, "
              f"prevalence={n_pos/max(n_rows,1):.4f}", flush=True)
    return np.vstack(X), np.concatenate(y)


class PlannerML(PlannerBase):
    """与 BALM 相同的逐分钟装箱，但预测器换成学习模型（冻结权重）。"""
    name = "Planner-Learned"

    def __init__(self, n_apps, mem_mb, budget_mb, cold_penalty=None, model=None,
                 mean_s=None, max_lag=120, **kw):
        super().__init__(n_apps, mem_mb, budget_mb, cold_penalty=cold_penalty, **kw)
        self.model = model
        self.cal_bins = 10
        # 快速路径：手写 sigmoid，避免每轮调用 sklearn 的 predict_proba（慢 10 倍以上）
        self.coef = np.asarray(model.coef_).reshape(-1)
        self.intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        self.max_lag = max_lag
        self.ft = FeatureTracker(n_apps, mem_mb, mean_s if mean_s is not None
                                 else np.full(n_apps, 0.5))
        self.last_cand_size = 0

    def set_aux(self, mean_s, cold_penalty):
        self.cold_penalty = cold_penalty
        self.mean_s = mean_s
        self.ft.mean_dur = np.log1p(mean_s.astype(np.float64) * 1000.0)

    def tick(self, t, active_apps):
        self.ft.tick(t, active_apps)

    def on_activity(self, t, active_apps, counts):
        self.ft.update_activity(t, active_apps)
        BasePolicy.on_activity(self, t, active_apps, counts)

    def replan(self, t, active_apps, counts):
        lag_all = t - self.ft.last_active
        # 与 BALM 完全相同的候选定义：1 <= lag <= G
        cand = np.nonzero((lag_all >= 1) & (lag_all <= self.max_lag))[0]
        self.last_cand_size = len(cand)
        if len(cand) == 0:
            self._apply(t, np.empty(0, dtype=np.int64))
            return
        F = self.ft.features(cand, t)
        z = F @ self.coef + self.intercept
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        # 供 sim.simulate 做与 BALM 相同的概率审计（Brier / AUROC / AUPRC）
        self._last_pred = p
        self._last_cand = cand
        self.cal_count = getattr(self, "cal_count", np.zeros(10, dtype=np.int64))
        self.cal_hit = getattr(self, "cal_hit", np.zeros(10, dtype=np.int64))
        self.brier_sum = getattr(self, "brier_sum", 0.0)
        self.brier_n = getattr(self, "brier_n", 0)
        value = p * self.cold_penalty[cand]
        self._apply(t, self._select_value(cand, value, t))

    def observe_outcome(self, t, active_apps):
        if getattr(self, "_last_pred", None) is None:
            return
        pred, cand = self._last_pred, self._last_cand
        hit = np.zeros(len(cand), dtype=np.float64)
        hit[np.isin(cand, active_apps)] = 1.0
        bins = np.clip((pred * self.cal_bins).astype(np.int64), 0, self.cal_bins - 1) \
            if hasattr(self, "cal_bins") else np.clip((pred * 10).astype(np.int64), 0, 9)
        np.add.at(self.cal_count, bins, 1)
        np.add.at(self.cal_hit, bins, hit.astype(np.int64))
        self.brier_sum += float(np.sum((pred - hit) ** 2))
        self.brier_n += len(cand)
        self._last_pred = None
        self._last_cand = None


def train_model(X, y, seed=0, class_weight=None):
    """逻辑回归（L2, lbfgs, max_iter=200）。

    默认不做 class re-weighting：训练样本取自与推理相同的候选集，类分布已一致，
    这样可以保留概率的校准性。"""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=200, C=1.0, class_weight=class_weight)
    m.fit(X, y)
    return m
