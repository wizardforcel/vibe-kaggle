# 第9章 时序比赛详解：Playground S5E1 贴纸销售预测（2025，时间纪律的入门标准）

> **本章目标**：入门级时序比赛的首要教学点不是模型复杂度，而是**时间验证纪律**。Top 15% 的公开方案使用 Group K-Fold 验证——说明时间/分组纪律从入门级比赛就是核心。

---

## 9.1 比赛与数据

预测不同国家、商店和产品的贴纸销售额，MAPE 指标，2025 年 Playground Series S5E1。

数据结构：每个 (国家, 商店, 产品, 日期) 一条记录，目标为销售额。强季节性（周、月、节假日）。

```mermaid
flowchart LR
    A[AutoGluon-TS/LGBM基线] --> B[人审各国/店误差]
    B --> C[人设定GroupKFold时间纪律]
    C --> D[Agent枚举滞后/滚动]
    D --> E[人判断周期含义]
    E --> F[集成+提交]
    style C fill:#fff9c4
    style E fill:#fff9c4
```

---

## 9.2 自动基线工件

AutoGluon-TimeSeries / LightGBM 基线工件；人检查不同国家/商店的误差模式。

```python
# src/baselines/s5e1_baseline.py
"""S5E1 基线：LightGBM + 日期特征，按国家/商店分组报告 MAPE。"""
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")

def add_date(df):
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["year"], df["month"], df["day"] = d.dt.year, d.dt.month, d.dt.day
    df["dow"], df["woy"] = d.dt.dayofweek, d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

# 训练基线并记录各国 MAPE（人审查用）
print("基线完成，按 country/store 分组报告 MAPE")
```

人审查：某国 MAPE 远高于其他 → 该国数据可能稀疏或有特殊节假日，需单独处理。

---

## 9.3 人设定 Group K-Fold 时间纪律

按商店和产品分组做时间切分，Agent 执行；**人检查工件中的折索引是否严格按时间顺序**。

为什么用 Group K-Fold？同一商店/产品的数据在时间上连续，若随机 KFold，同一商店的相邻日期会同时出现在训练/验证，造成泄漏。GroupKFold 按商店分组，确保同组不跨折。

```python
# src/cv/group_time_cv.py
"""
Group + 时间双重纪律：先按商店分组，再在时间上切分。
简化实现：按 (store, 时间段) 分块，确保验证块时间晚于训练块。
"""
import numpy as np, pandas as pd

def group_time_splits(df, group_col, time_col, n_splits=5):
    """按组做前向时间切分：每组内按时间排序，前 80% 训练，后 20% 验证。"""
    splits = []
    groups = df[group_col].unique()
    # 全局时间分块，确保验证时间 > 训练时间
    times = np.sort(df[time_col].unique())
    blocks = np.array_split(times, n_splits + 1)
    for k in range(n_splits):
        tr_times = np.concatenate(blocks[:k+1])
        va_times = blocks[k+1]
        tr = df[df[time_col].isin(tr_times)].index.values
        va = df[df[time_col].isin(va_times)].index.values
        splits.append((tr, va))
    return splits
```

> Top 15% 的公开方案使用 Group K-Fold 验证，说明时间/分组纪律从入门级比赛就是核心。

---

## 9.4 Agent 枚举滞后与滚动特征，人判断周期含义

Agent 批量生成滞后 1/7/14/28 天和滚动窗口的工件；人根据零售数据的周周期和月度效应判断哪些窗口有业务含义。

```python
# src/features/s5e1_features.py
"""S5E1 特征：滞后 + 滚动 + 日期 + 节假日。"""
import numpy as np, pandas as pd

def make_features(df, group_cols=["country", "store", "product"]):
    """按 (country,store,product) 分组做滞后/滚动。"""
    df = df.sort_values(group_cols + ["date"]).copy()
    g = df.groupby(group_cols)["sales"]
    for k in [1, 7, 14, 28]:
        df[f"lag_{k}"] = g.shift(k)
    for w in [7, 14, 28]:
        r = g.shift(1).rolling(w, min_periods=1)
        df[f"roll_mean_{w}"] = r.mean()
        df[f"roll_std_{w}"] = r.std()
    return df
```

**人判断**：
- 滞后 7 天：周周期，零售核心信号；
- 滞后 28 天：月周期，月度补货效应；
- 滚动 7 天均值：短期趋势；
- 滞后 3 天：无明显业务含义，可作噪声对照。

---

## 9.5 集成与提交

```python
# src/submit/s5e1_submit.py
"""多工件融合 + 提交核对。MAPE 指标，注意预测非负。"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
def fuse_and_submit(paths, sample_path, out_path, target="sales", id="id"):
    preds = [np.load(p) for p in paths]
    final = np.mean(preds, axis=0)
    final = np.clip(final, 0, None)          # 销售额非负
    sample = pd.read_csv(sample_path)
    sub = sample.copy(); sub[target] = final
    sub.to_csv(out_path, index=False)
    assert (sub[target] >= 0).all()
    print(f"✅ 提交通过: {out_path}")
```

提交前检查：预测非负、ID 顺序、各国 MAPE 无异常。

---

## 9.6 小结

**入门级时序比赛的首要教学点不是模型复杂度，而是时间验证纪律。**

- Group K-Fold 防同组泄漏；
- 滞后/滚动窗口的业务含义由人判断；
- 提交前非负约束断言。

> **下一章预告**：第 10 章道路事故风险（时序视角），展示 Agent 枚举模型、人做互补性选择；第 11 章 Jane Street 高频金融，验证纪律的极限。