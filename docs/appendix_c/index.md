# 附录C：AutoGluon 快速参考（Tabular / TimeSeries）

> AutoGluon 是表格和时序 AutoML 的主力框架，能几行代码跑出强基线，全书多次作为“第一个合规工件”的来源。

---

## C.1 Tabular 预测器

```python
from autogluon.tabular import TabularPredictor

# 初始化
predictor = TabularPredictor(
    label="target",              # 目标列
    eval_metric="auc",           # 指标：auc / logloss / rmse / mape ...
    path="ag_models",            # 模型保存目录
)

# 训练（自动选模型、预处理、stacking）
predictor.fit(
    train_data=train_df,
    time_limit=300,              # 秒
    presets="medium_quality",    # medium / good_quality / best_quality
    verbosity=2,
)

# 预测
pred = predictor.predict(test_df)              # 回归/类别
pred_proba = predictor.predict_proba(test_df)  # 分类概率，取正类列
```

**presets 常用值**：
- `medium_quality`：快，适合原型；
- `good_quality`：稍慢，更强；
- `best_quality`：慢，最强（多层 stacking）；
- `best_v150`：更快更强（v1.5+）；
- `extreme`：需要 GPU，数据 <10 万样本时最强（内置 TabPFN 等表格基础模型）。

**表格基础模型（TabPFN 等）**：需 `pip install autogluon.tabular[tabarena]`，预设 `extreme` / `noncommercial` 会自动使用。

---

## C.2 TimeSeries 预测器

```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    target="target",
    prediction_length=28,        # 预测未来 28 步
    eval_metric="MAPE",
    path="ag_ts_models",
)
predictor.fit(train_data=ts_df, time_limit=300, verbosity=0)
pred = predictor.predict(ts_df)  # 返回 item_id × timestamp × 目标
```

**时序数据格式**：长表，至少包含 `item_id`、`timestamp`、`target` 三列。

---

## C.3 产出合规工件（全书规范）

AutoGluon 内部有 OOF，但全书要求用**人定的固定折**产出 OOF，便于集成。示例：

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ART = Path("artifacts/oof"); ART.mkdir(parents=True, exist_ok=True)

def autogluon_fixed_cv_oof(train, test, splits, target="target", id="id"):
    """按固定 splits 做 OOF，存 .npy 工件。"""
    oof = np.zeros(len(train), dtype=np.float32)
    for k, (tr, va) in enumerate(splits):
        p = TabularPredictor(label=target, eval_metric="auc",
                             path=str(ART / f"_fold{k}"))
        p.fit(train.iloc[tr], time_limit=120, presets="medium_quality", verbosity=0)
        pr = p.predict_proba(train.iloc[va].drop(columns=[target, id], errors="ignore"))
        oof[va] = pr.iloc[:, 1].values
    np.save(ART / "train_oof_AUTOGLUON_v001.npy", oof)
    # 测试集全量训练
    p_full = TabularPredictor(label=target, eval_metric="auc", path=str(ART / "_full"))
    p_full.fit(train, time_limit=300, presets="medium_quality", verbosity=0)
    tp = p_full.predict_proba(test.drop(columns=[id], errors="ignore")).iloc[:, 1].values
    np.save(ART / "test_preds_AUTOGLUON_v001.npy", tp.astype(np.float32))
```

下一附录是 PyTorch / HuggingFace 速查。