# 第9章 时序比赛详解：Playground S5E1 贴纸销售预测（2025，时间纪律的入门标准）

> **本章目标**：把第 8 章的时序循环，落到一场入门级比赛上。你会看到 Top 15% 的公开方案都在用 Group K-Fold 验证——**时间/分组纪律不是高级技巧，而是入门级比赛的核心。** 这一章教你“为什么不能随机切，要按组切”。

---

## 开篇：同一家店的销量，随机切开会“泄漏”

第 8 章我们说了，时序不能随机切分，要按时间顺序。但 S5E1 贴纸销售有个额外陷阱：**同一个商店（或同一家店卖的同款贴纸）的数据，在时间上是连续的**。

想象你要预测某家贴纸店的日销量。随机 KFold 切分时，可能出现：1 月 10 日在训练集、1 月 11 日在验证集。但这两天的销量高度相关（同一店、连续两天）。模型在“未来”（1 月 10 日）训练，去预测“过去”（1 月 11 日）——又是偷看未来。

更隐蔽的是**同店泄漏**：同一商店的所有日期被随机撒到各折里，模型能“记住”某家店的特征，跨折作弊。

解法就是这一章的主角：**Group K-Fold**——按“商店 + 时间段”分组，同组不跨折。

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

## 9.1 比赛与数据

预测不同国家、商店和产品的贴纸销售额，指标 MAPE（平均绝对百分比误差），2025 年 Playground Series S5E1。

数据结构：每个（国家, 商店, 产品, 日期）一条记录，目标是销售额。**强季节性**：周（周末高）、月（月初月末）、节假日。

MAPE 是什么？它衡量“预测偏差占真实值的百分比”的平均。比如真实销量 100，你预测 110，偏差 10%，那这条就贡献 10%。MAPE 的好处是“对量级不敏感”——卖 100 的和卖 10000 的店，都用百分比衡量误差。

---

## 9.2 自动基线工件

AutoGluon-TimeSeries / LightGBM 基线工件；人检查不同国家/商店的误差模式。

```python
# src/baselines/s5e1_baseline.py
"""S5E1 基线：LightGBM + 日期特征，按国家/商店分组报告 MAPE。"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
train = pd.read_csv(ROOT / "data" / "train.csv")
test = pd.read_csv(ROOT / "data" / "test.csv")


def add_date(df):
    """拆日期特征。"""
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek                          # 星期几（0=周一）
    df["woy"] = d.dt.isocalendar().week.astype(int)     # 一年第几周
    df["is_weekend"] = (df["dow"] >= 5).astype(int)     # 是否周末
    return df
```

跑完基线，人分组看 MAPE：

```python
def mape_by_group(oof_df, y, group_col):
    """按某列分组报告 MAPE，找出误差异常的组。"""
    errs = np.abs(oof_df - y) / np.clip(y, 1e-8, None)   # 绝对百分比误差
    out = pd.DataFrame({"group": group_col, "err": errs})
    return out.groupby("group")["err"].mean().sort_values(ascending=False)
```

人审查：某国 MAPE 远高于其他 → 该国数据可能稀疏或节假日特殊，需单独处理。**这就是“人审查误差模式”的具体动作**——不是一眼看总数，而是拆开看每个组。

---

## 9.3 人设定 Group K-Fold 时间纪律

按商店和产品分组做时间切分，Agent 执行；**人检查工件中的折索引是否严格按时间顺序**。

简化版的“分组 + 时间”双重切分，代码这样写：

```python
# src/cv/group_time_cv.py
"""
Group + 时间双重纪律：
先按 (store, product) 分组确保同组不跨折，
再在时间上切分确保验证时间晚于训练时间。
"""
import numpy as np
import pandas as pd


def group_time_splits(df, time_col, n_splits=5):
    """
    全局时间分块 + 前向切分。
    把时间轴切成 n_splits+1 块，第 k 折用前 k+1 块训练、第 k+2 块验证。
    这样每折的训练时间都严格早于验证时间。
    """
    splits = []
    times = np.sort(df[time_col].unique())
    blocks = np.array_split(times, n_splits + 1)
    for k in range(n_splits):
        tr_times = np.concatenate(blocks[: k + 1])      # 训练：前 k+1 块
        va_times = blocks[k + 1]                          # 验证：下一块
        tr = df.index[df[time_col].isin(tr_times)].values
        va = df.index[df[time_col].isin(va_times)].values
        splits.append((tr, va))
    return splits
```

> Top 15% 的公开方案使用 Group K-Fold 验证，说明时间/分组纪律从入门级比赛就是核心。这不是“锦上添花”，而是“不这么做就必输”。

**人检查**：把折索引算出来，过一遍 `check_time_order`（第 8 章那个函数），确认每折验证时间都晚于训练时间。

---

## 9.4 Agent 枚举滞后与滚动特征，人判断周期含义

Agent 批量生成滞后 1/7/14/28 天和滚动窗口的工件；人根据零售数据的周周期和月度效应判断哪些窗口有业务含义。

```python
# src/features/s5e1_features.py
"""S5E1 特征：滞后 + 滚动 + 日期 + 节假日。"""
import numpy as np
import pandas as pd


def make_features(df, group_cols=["country", "store", "product"]):
    """按 (country,store,product) 分组做滞后/滚动。"""
    df = df.sort_values(group_cols + ["date"]).copy()
    g = df.groupby(group_cols)["sales"]
    for k in [1, 7, 14, 28]:
        df[f"lag_{k}"] = g.shift(k)              # 滞后 k 天
    for w in [7, 14, 28]:
        r = g.shift(1).rolling(w, min_periods=1)  # shift(1) 防泄漏
        df[f"roll_mean_{w}"] = r.mean()
        df[f"roll_std_{w}"] = r.std()
    return df
```

**人判断**（这是 Agent 干不了的）：
- 滞后 7 天：周周期，零售核心信号；
- 滞后 28 天：月周期，月度补货效应；
- 滚动 7 天均值：短期趋势；
- 滞后 3 天：无明显业务含义，可作噪声对照。

判断依据是“零售数据是怎么生成的”——你知道周末促销、月末盘点这些业务规律，Agent 不知道。

---

## 9.5 集成与提交

```python
# src/submit/s5e1_submit.py
"""多工件融合 + 提交核对。MAPE 指标，注意预测非负。"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fuse_and_submit(paths, sample_path, out_path, target="sales", id_col="id"):
    """把多个工件按简单平均融合，再核对非负。"""
    preds = [np.load(p) for p in paths]
    final = np.mean(preds, axis=0)
    final = np.clip(final, 0, None)            # 销售额不能为负！
    sample = pd.read_csv(sample_path)
    sub = sample.copy()
    sub[target] = final
    sub.to_csv(out_path, index=False)
    assert (sub[target] >= 0).all()
    print(f"✅ 提交通过：{out_path}")
```

那个 `np.clip(..., 0, None)` 别省。回归模型很容易预测出负数，但销售额不可能是负数。提交一个负数，MAPE 会算出荒谬的值。**提交前断言非负，是时序回归的标配动作。**

---

## 9.6 小结

**入门级时序比赛的首要教学点不是模型复杂度，而是时间验证纪律。**

- Group K-Fold 防同组泄漏 + 时间切分防未来泄漏；
- 滞后/滚动窗口的业务含义由人判断；
- `shift(1)` 防当天泄漏；
- 提交前非负约束断言。

这一章最有价值的认知是：**时序入门的第一道坎，不是“用什么模型”，而是“怎么切数据”。** 很多人卡在分数上，其实是在切分方式上埋了雷。

> **下一章预告**：第 10 章回到表格与时间交叉的 S5E10（时序视角），看 Agent 怎么在时间切分下管理实验、人怎么挑互补子集；第 11 章则是时序的极限——Jane Street 高频金融数据，连“验证本身”都要怀疑。