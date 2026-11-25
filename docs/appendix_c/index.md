# 附录C：AutoGluon 快速参考（Tabular / TimeSeries）

> 本附录是全书"自动基线"的主力工具。定位只有一条：**只保留产出合规工件所需的调用，以及必须人来判断的那几个开关**。AutoGluon 能帮你跑几十个模型，但"用哪个 preset、给多少时间、结果可不可信"——这些得人拍板。

先看全书表格章统一的基线调用模板：

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label="target",              # 目标列
    eval_metric="auc",           # 二分类 auc，回归 rmse/mae，时序 mape
    path="artifacts/models/",    # 工件落盘目录
)

predictor.fit(
    train_data=train_df,
    presets="medium_quality",    # 档位，见 C.2
    time_limit=600,              # 秒，见 C.2
    num_bag_folds=5,             # bagging 折数
    num_stack_levels=0,          # 人工 stack 的话先设 0，避免重复
)

# 产出合规工件（见 C.3）
oof = predictor.predict(train_df)           # 注意：这里是训练集预测，OOF 需按固定折另算
test_pred = predictor.predict(test_df)
predictor.leaderboard(silent=False)          # 看内部集成结果
```

`predictor.fit` 返回的就是训练好的模型，`path` 指定的目录会自动落盘模型文件。注意：AutoGluon 内部做的 bagging/stacking 是**它自己的折**，和全书第 2 章要求的"人固定验证协议"不是同一套折——所以产出 OOF 工件时，要按 C.3 的标准写法另算。

---

## C.1 Tabular 快速开始

TabularPredictor 是表格赛的默认入口。最小可用版本只有三行：

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

predictor = TabularPredictor(label="target", eval_metric="auc")
predictor.fit(train, presets="medium_quality", time_limit=600)

pred = predictor.predict(test)
submission = test[["id"]].copy()
submission["target"] = pred
submission.to_csv("submission.csv", index=False)
```

注意 `eval_metric` 要和比赛指标对齐：二分类用 `auc`，回归用 `rmse`/`mae`，时序用 `mape`。选错指标，模型优化的方向就全错了（这也是全书反复强调的"指标由人定"）。

**什么时候会用到**：每一章表格赛的第一步基线；想快速知道"这条数据上不费力能做到多好"时。

---

## C.2 presets 档位与 time_limit 的取舍

AutoGluon 提供四个 preset 档位，控制模型家族数量、bagging 层数、stacking 层数：

| 档位 | 特点 | 适用 |
|------|------|------|
| `fast` | 少数模型、少 bagging、不 stacking | 快速探路、大数据先过一遍 |
| `medium_quality` | 默认平衡 | 大部分竞赛的起点 |
| `high_quality` | 更多模型、更多 bagging | 有时间、想多榨一点分数 |
| `best_quality` | 全模型家族、多层 stacking | 时间充裕、冲榜前列 |

`time_limit` 是秒级的总时限。它控制"最多跑多久"，但**不是"保证跑满这么久"**——模型会在时间用完或内部条件满足时提前停。

两个容易混淆的点：
- `time_limit` 设太小，基线可能还没训完就结束，分数虚低；建议第一次设宽松一点（比如 1800 秒 = 30 分钟），再看 leaderboard 里各模型耗时。
- `num_bag_folds` / `num_stack_levels` 决定内部集成深度，默认 `medium_quality` 已经开启；如果要人工做 stacking（C.4），这里先设 `num_stack_levels=0`，避免重复嵌套。

**什么时候会用到**：第一次跑一条数据时用 `fast` 探路；确定方向后用 `medium_quality` 产出合规基线工件；冲榜前用 `high_quality` 再榨一次。

---

## C.3 产出 OOF 与测试预测工件的标准写法

AutoGluon 内部做的 bagging 是基于它自己切的折，和全书第 2 章"人固定验证协议"不是同一套折。所以**不要直接把 `predictor.predict(train_df)` 当 OOF 工件用**。合规的 OOps 必须按人定的固定折另算：

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

def make_compliant_oof(train_df, label="target", n_folds=5, seed=42):
    """
    按人固定的 StratifiedKFold 产出 OOF 工件。
    返回形状 (n_train,) 的 OOF 预测数组。
    """
    oof = np.zeros(len(train_df), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[label])):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        # 每一折单独训一个模型，只用验证折做 predict
        p = TabularPredictor(label=label, eval_metric="auc")
        p.fit(fold_train, presets="medium_quality", time_limit=300)
        oof[val_idx] = p.predict(fold_val).values

    return oof

oof = make_compliant_oof(train_df)
np.save("artifacts/oof/train_oof_AUTOMGluon_v001.npy", oof)
```

关键点：
- 每一折单独 `fit`，验证折只用来 `predict`，绝不把验证折的数据混进训练；
- 折索引固定（`random_state=42`），后续所有模型用同一套折，OOF 才可比较；
- 结果按 `train_oof_{模型}_{版本}.npy` 规范落盘，与 `test_preds` 一起构成合规工件。

测试预测则直接用全量训练集训一个模型，对 test 做 `predict`，存 `test_preds_{模型}_{版本}.npy`。

**什么时候会用到**：每一章表格赛的"第 3.1 自动基线 + 工件落盘"，是全书表格流程的第一个合规工件。

---

## C.4 内部集成机制与"它已经 stacking 过了"

AutoGluon 自带多层 bagging + stacking：它会训很多个模型，再用它们的 OOF 做元模型组合。所以`predictor.leaderboard()` 里的最佳模型，本质上已经是"内部集成"的结果。

这对人工 stacking（第 3.5/21 章）有个重要含义：**不要重复造 AutoGluon 已经做好的那层。** 人工 stacking 的价值在于引入 AutoGluon 之外的模型家族（TabPFN、神经网络、手工特征模型），把它们的 OOF 当成新特征再做一次集成。

简单记法：AutoGluon 负责"模型家族内部的 stack"；人负责"跨家族的 stack"。

**什么时候会用到**：决定人工集成范围时；看 leaderboard 时理解哪些分数是"内部集成带来的"。

---

## C.5 TimeSeries 快速开始

```python
from autogluon.timeseries import TimeSeriesPredictor
import pandas as pd

# 长格式数据：item_id、timestamp、target
train = pd.read_csv("data/train.csv")

predictor = TimeSeriesPredictor(
    target="target",
    prediction_length=28,      # 预测未来 28 个时点
    eval_metric="MAPE",
    path="artifacts/models/",
)

predictor.fit(train, presets="medium_quality", time_limit=600)
pred = predictor.predict(train)   # 返回 long 格式的预测
```

时序数据必须是**长格式**：每一行是一个（序列、时间点）的观测，列里要有 `item_id`（序列编号）、`timestamp`（时间）、`target`（值）。宽格式（每列一个序列）需要先 `melt` 成长格式。

官方 API 要求每个序列足够长，能切出内部验证集；如果某序列太短，会自动跳过或报错，跑不完时要检查序列长度分布。

**什么时候会用到**：第 8–11 章时序赛的自动基线；和零样本基线（Chronos / TimesFM）并列做对照工件。

---

## C.6 时序基线工件与零样本对照

时序赛里，AutoGluon-TimeSeries 基线不应单独决策，要和零样本基线并列：

- AutoGluon-TimeSeries：训练后的时序模型，有你的数据特定知识；
- Chronos / TimesFM：零样本预测，没见过你的数据，直接外推。

两者误差模式不同，是后续融合（第 8.4/12.4 节）的多样性来源。把两者的预测都按合规工件规范落盘，再按人定的时间折做集成。

**什么时候会用到**：时序章的"8.1 时序自动基线 + 工件落盘"，作为表格基线在时序上的对照。

---

## C.7 常见坑

1. **内存不足**：AutoGluon 会训几十个模型，内存容易爆。降 presets（`fast` → `medium_quality`）、减 `num_bag_folds`，或缩小 `time_limit`。
2. **时间预算被第一个模型吃光**：如果第一个模型就跑了 20 分钟，后面来不及了。先用 `fast` 探路，看各模型耗时，再分配总预算。
3. **内部折与外部 CV 折不一致**：AutoGluon 内部切的折和你人固定的折不同，OOF 不可比。必须按 C.3 按人固定折另算，不要直接拿 `predict(train_df)` 当 OOF。
4. **GPU 不可用时自动回退 CPU**：AutoGluon 会静默降级，训练变慢但不会报错。跑前先确认环境里是否有 GPU，否则预期时间要翻倍。

**什么时候会用到**：跑基线时遇到 OOM / 耗时过长 / OOF 可比性存疑，对照这一条排查。

跑完基线最常见的困惑是"分数对不上"，可以这样问：

> **提示词**：AutoGluon 的 leaderboard 显示最佳模型 CV 0.91，但我自己按 5 折重算 OOF 只有 0.88，差好多。这是哪里出了问题？只给排查方向。

模型会这样答：

> **AI 输出**
>
> （思考过程）用户要排查方向，不要我改代码。差异来源通常是"折不一样"
> 和"内部集成"。按这两点组织，再补一个"别慌"的判断。
>
> 两个最可能的原因：
>
> 1. **折不一样**：AutoGluon 内部用自己的 bagging 折，你用的是人固定的
>    StratifiedKFold。两套折的验证集不是同一批样本，分数天然不可比。
>    → 这是正常现象，不是 bug。你要的 OOF 必须按你自己固定的折算。
> 2. **内部集成**：leaderboard 上的最佳分数是"内部 stacking 之后"的结果，
>    你单模型重算自然低一截。
>    → 也是正常的。人工集成要在此基础上再加"跨家族"的模型。
>
> 判断标准：**只要你的 OOF 是在固定折上算的、折索引哈希一致，它就是合规的**，
> 不需要和 leaderboard 对齐。两者回答的是不同问题。

---

## 小结

AutoGluon 是全书表格章的"自动基线"主力：

- 入口是 `TabularPredictor` / `TimeSeriesPredictor`，三行代码就能跑；
- `presets` 控制模型家族数量，`time_limit` 控制总时长，**人来拍板**；
- OOF 必须按**人固定的折**另算，按 `train_oof_[模型]_[版本].npy` 规范落盘；
- AutoGluon 做的是"模型家族内部集成"，人工 stack 的价值在"跨家族"。

下一个附录是深度学习底座：PyTorch 与 HuggingFace Transformers，重点是"冻结骨干 + 轻量头"这条图像/文本章的主路径。