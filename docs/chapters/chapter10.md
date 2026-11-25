# 第10章 时序比赛详解：Playground S5E10 道路事故风险预测（2025，Agent 枚举与人做互补性选择）

> **本章目标**：这场比赛同时具有表格和时序特征，放在时序部分侧重**时间切分下的 Agent 实验管理**。核心：Agent 扩大模型候选池，人的判断集中在“互补性”而非“单模型最高分”。

---

## 10.1 比赛与数据

回归任务，RMSE；数据包含时间相关的道路/环境特征。

与第 5 章同一场比赛，但本章从时序视角切入：数据含时间特征（月份、星期、节假日），需考虑时间切分。

```mermaid
flowchart LR
    A[时间切分下基线] --> B[记录随机CV vs 时间CV差异]
    B --> C[Agent枚举XGB/TabM/LGBM/NN]
    C --> D[OOF相关性矩阵]
    D --> E[Greedy NNLS选互补]
    E --> F[人确认子集]
    style B fill:#fff9c4
    style E fill:#fff9c4
```

---

## 10.2 自动基线在时间切分下的工件

AutoGluon 基线在时间切分下的 CV；与随机 CV 的分数差异由人记录和解读。

```python
# src/baselines/s5e10_time_cv.py
"""对比随机 CV 与时间 CV 的 RMSE 差异，判断时间泄漏风险。"""
import numpy as np, pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
train = pd.read_csv(ROOT / "data" / "train.csv")
TARGET = "target"

def cv_rmse(splits):
    oof = np.zeros(len(train))
    for tr, va in splits:
        p = TabularPredictor(label=TARGET, eval_metric="rmse",
                             path=str(ROOT / "artifacts" / "_tmp")).fit(
            train.iloc[tr], time_limit=60, verbosity=0)
        oof[va] = p.predict(train.iloc[va]).values
    return mean_squared_error(train[TARGET], oof, squared=False)

random_rmse = cv_rmse(list(KFold(5, shuffle=True, random_state=42).split(train)))
time_rmse = cv_rmse(list(TimeSeriesSplit(5).split(train)))
print(f"随机CV RMSE={random_rmse:.4f}, 时间CV RMSE={time_rmse:.4f}")
print(f"差异 {time_rmse - random_rmse:+.4f}（若时间CV明显更差，说明存在时间泄漏）")
```

**人解读**：若时间 CV 明显差于随机 CV，说明模型依赖了时间相关的伪信号（如用未来信息预测过去），需修正特征。

---

## 10.3 Agent 枚举模型，人选择互补子集

Agent 生成 XGBoost、TabM、LightGBM、NN 的工件；人通过 OOF 相关性矩阵和 Greedy NNLS 判断哪些模型值得进入集成，**而不是让 Agent 自动做 hill climbing 后直接提交**。

```python
# src/ensembles/complement_select.py
"""
互补性选择：相关性矩阵 + Greedy NNLS。
与第5章类似，但强调时间切分下 OOF 的相关性结构。
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error

def select_complementary(oofs, y, max_k=6):
    """先聚类去冗余，再 NNLS 选互补。"""
    names = list(oofs)
    M = np.column_stack([oofs[n] for n in names])
    corr = np.corrcoef(M, rowvar=False)
    Z = linkage(1 - corr, method="average")
    labels = fcluster(Z, t=0.02, criterion="distance")
    # 每簇留一个代表（取簇内 CV 最优，这里简化为第一个）
    reps = {}
    for n, l in zip(names, labels):
        reps.setdefault(l, n)
    cand = list(reps.values())
    # NNLS 贪心
    chosen, cur = [], np.inf
    while len(chosen) < max_k:
        best, best_r, best_w = None, cur, None
        for n in cand:
            if n in chosen: continue
            c = chosen + [n]
            MM = np.column_stack([oofs[m] for m in c])
            w, _ = nnls(MM, y)
            r = mean_squared_error(y, MM @ w, squared=False)
            if r < best_r - 1e-5:
                best, best_r, best_w = n, r, w
        if best is None: break
        chosen.append(best); cur = best_r
    return chosen
```

> 人的判断集中在“互补性”而非“单模型最高分”——这是本章与第 5 章呼应的核心。

---

## 10.4 提交

```python
# 提交文件生成 + 时间切分验证记录
from src.submit.make_submission import make_submission
make_submission("artifacts/test_preds/test_preds_NNLS_v001.npy",
                "data/sample_submission.csv",
                "artifacts/submissions/submission_NNLS_v001.csv")
# 同时在日志记录时间CV与随机CV差异，便于复盘
```

---

## 10.5 小结

**Agent 扩大模型候选池，人的判断集中在“互补性”而非“单模型最高分”。**

- 时间切分 vs 随机 CV 差异是泄漏信号；
- 相关性聚类去冗余 + NNLS 选互补；
- 提交记录时间验证信息。

> **下一章预告**：第 11 章 Jane Street 高频金融，验证本身极其困难时，人的判断是“这个分数是否可信”。