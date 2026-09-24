#!/usr/bin/env python3
"""
00_build_dataset.py
将 Azure Functions 2019 trace 转换为仿真用的紧凑数据集。

输入（原始 trace，见 data/download_trace.sh）:
  invocations_per_function_md.anon.d[01-14].csv   每个函数每分钟调用次数
  function_durations_percentiles.anon.d[01-14].csv 函数执行耗时分布(ms)
  app_memory_percentiles.anon.d[01-12].csv         应用内存分配(MB)

输出:
  data/derived/trace.npz
     counts_csr  : (n_apps x 20160) 每分钟调用次数, 稀疏
     app_mem_mb  : 每个应用的平均内存占用 (MB)
     app_p50_ms  : 每个应用的调用加权中位执行耗时 (ms)
     app_mean_ms : 每个应用调用加权的平均执行耗时 (ms)
     app_q_ms    : (n_apps x 8) 调用加权的耗时分位点 (ms), 用于采样
     app_q_w     : 每个分位点的累计权重 (用于逆变换采样)
     app_trigger : 主导触发器类型的编码
     trigger_names
     minute_of_day / day_of_trace 辅助索引
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse

RAW = os.environ.get("AZ_TRACE_DIR", "/tmp/aztrace/data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "derived")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS_INV = range(1, 15)   # 14 天调用
DAYS_DUR = range(1, 15)   # 14 天耗时
DAYS_MEM = range(1, 13)   # 12 天内存
MINUTES_PER_DAY = 1440
TOTAL_MINUTES = 14 * MINUTES_PER_DAY

TRIGGERS = ["http", "timer", "event", "queue", "storage", "orchestration", "others"]
TRIG_IDX = {t: i for i, t in enumerate(TRIGGERS)}


def minute_cols(df):
    return [str(i) for i in range(1, MINUTES_PER_DAY + 1)]


def load_invocations():
    """返回 (按应用聚合的每分钟调用次数 DataFrame, 触发器等元数据)."""
    per_day = []
    trig_counter = {}
    for d in DAYS_INV:
        path = os.path.join(RAW, f"invocations_per_function_md.anon.d{d:02d}.csv")
        df = pd.read_csv(path)
        cols = minute_cols(df)
        # 应用级聚合：同一应用下所有函数的调用次数求和
        agg = df.groupby("HashApp", sort=False)[cols].sum()
        per_day.append(agg)
        # 记录应用主导触发器（取该应用下函数数最多的触发器组）
        tc = df.groupby(["HashApp", "Trigger"]).size().reset_index(name="n")
        for app, trig in zip(*[tc[c].values for c in ("HashApp", "Trigger")]):
            trig_counter.setdefault(app, {})
            trig_counter[app][trig] = trig_counter[app].get(trig, 0)
        del df
        print(f"  invocations day {d:02d}: {agg.shape[0]} apps, "
              f"{int(agg.to_numpy().sum())} invocations", flush=True)
    return per_day, trig_counter


def load_durations():
    """函数耗时分位点 -> 应用级调用加权分位点."""
    pct_cols = ["percentile_Average_0", "percentile_Average_1", "percentile_Average_25",
                "percentile_Average_50", "percentile_Average_75", "percentile_Average_99",
                "percentile_Average_100"]
    frames = []
    for d in DAYS_DUR:
        path = os.path.join(RAW, f"function_durations_percentiles.anon.d{d:02d}.csv")
        df = pd.read_csv(path, usecols=["HashApp", "Count", "Average"] + pct_cols)
        frames.append(df)
    dur = pd.concat(frames, ignore_index=True)
    # 清理：负值（日志错误）与异常 Count
    for c in pct_cols + ["Average"]:
        dur[c] = dur[c].clip(lower=0.0)
    dur["Count"] = dur["Count"].clip(lower=0.0)
    dur = dur[dur["Count"] > 0]
    return dur, pct_cols


def load_memory():
    frames = []
    for d in DAYS_MEM:
        path = os.path.join(RAW, f"app_memory_percentiles.anon.d{d:02d}.csv")
        df = pd.read_csv(path, usecols=["HashApp", "SampleCount", "AverageAllocatedMb"])
        frames.append(df)
    mem = pd.concat(frames, ignore_index=True)
    mem["AverageAllocatedMb"] = mem["AverageAllocatedMb"].clip(lower=1.0)
    # 按 SampleCount 加权平均，得到应用级内存占用
    w = mem.groupby("HashApp")["SampleCount"].sum()
    m = mem.groupby("HashApp").apply(
        lambda g: np.average(g["AverageAllocatedMb"], weights=g["SampleCount"])
    )
    return pd.DataFrame({"HashApp": m.index, "mem_mb": m.values})


def main():
    print("== 读取调用数据 ==")
    per_day, trig_counter = load_invocations()

    all_apps = sorted(set().union(*[set(a.index) for a in per_day]))
    app_index = {a: i for i, a in enumerate(all_apps)}
    n_apps = len(all_apps)
    print(f"应用总数: {n_apps}")

    print("== 构建分钟级稀疏矩阵 ==")
    rows, cols, vals = [], [], []
    for di, agg in enumerate(per_day):
        offset = di * MINUTES_PER_DAY
        sub = agg.to_numpy(dtype=np.int32)
        apps = agg.index
        # 只保留非零项
        nz = np.nonzero(sub)
        rows.append(np.array([app_index[a] for a in apps], dtype=np.int32)[nz[0]])
        cols.append((nz[1] + offset).astype(np.int32))
        vals.append(sub[nz])
        del sub
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals).astype(np.int32)
    counts = sparse.csr_matrix((vals, (rows, cols)), shape=(n_apps, TOTAL_MINUTES),
                               dtype=np.int32)
    counts.sum_duplicates()
    print(f"稀疏矩阵: {counts.shape}, 非零 {counts.nnz:,}, "
          f"总调用 {int(counts.sum()):,}")

    print("== 读取内存与耗时 ==")
    mem = load_memory()
    dur, pct_cols = load_durations()

    mem_map = dict(zip(mem.HashApp, mem.mem_mb))
    app_mem_mb = np.array([mem_map.get(a, np.nan) for a in all_apps], dtype=np.float32)

    # 应用级耗时：按 Count 加权
    grp = dur.groupby("HashApp")
    app_mean = grp.apply(lambda g: np.average(g["Average"], weights=g["Count"]))
    mean_map = dict(app_mean)
    app_mean_ms = np.array([mean_map.get(a, np.nan) for a in all_apps], dtype=np.float32)

    # 应用级分位点：对每个分位点做 Count 加权平均（近似），单调化后作为经验分布
    q_mats = {}
    for c in pct_cols:
        app_q = grp.apply(lambda g, cc=c: np.average(g[cc], weights=g["Count"]))
        q_mats[c] = dict(app_q)
    app_q_ms = np.zeros((n_apps, len(pct_cols)), dtype=np.float32)
    for j, c in enumerate(pct_cols):
        for i, a in enumerate(all_apps):
            app_q_ms[i, j] = q_mats[c].get(a, np.nan)
    # 单调化（分位点必须非降）并填充缺失
    app_q_ms = np.sort(np.nan_to_num(app_q_ms, nan=0.0), axis=1)
    app_q_w = np.array([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0], dtype=np.float32)

    # 主导触发器
    app_trigger = np.zeros(n_apps, dtype=np.int8)
    for i, a in enumerate(all_apps):
        tc = trig_counter.get(a)
        if tc:
            trig = max(tc.items(), key=lambda kv: kv[1])[0]
            app_trigger[i] = TRIG_IDX.get(trig, 6)

    out = os.path.join(OUT_DIR, "trace.npz")
    np.savez_compressed(
        out,
        counts_data=counts.data.astype(np.int32),
        counts_indices=counts.indices.astype(np.int32),
        counts_indptr=counts.indptr.astype(np.int64),
        counts_shape=np.array(counts.shape, dtype=np.int64),
        app_mem_mb=app_mem_mb,
        app_mean_ms=app_mean_ms,
        app_q_ms=app_q_ms,
        app_q_w=app_q_w,
        app_trigger=app_trigger,
        trigger_names=np.array(TRIGGERS),
        total_minutes=np.array([TOTAL_MINUTES]),
        minutes_per_day=np.array([MINUTES_PER_DAY]),
    )
    print(f"写出 {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"内存覆盖率: {np.isfinite(app_mem_mb).mean():.1%}, "
          f"耗时覆盖率: {np.isfinite(app_mean_ms).mean():.1%}")


if __name__ == "__main__":
    sys.exit(main())
