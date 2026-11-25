# 附录B：scikit-learn / XGBoost / LightGBM 速查

> 本附录是三个最常用机器学习库的速查。scikit-learn 是通用工具库，XGBoost 和 LightGBM 是表格竞赛的两大主力梯度提升库。

---

## B.1 scikit-learn 速查

```python
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, r2_score

# ---- 切分 ----
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

# 5 折分层（分类）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in skf.split(X, y):
    ...

# 时序切分（时序）
tscv = TimeSeriesSplit(n_splits=5, gap=28)

# ---- 预处理（务必只在训练折 fit！） ----
scaler = RobustScaler().fit(X_tr)        # 稳健标准化，抗异常值
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)          # 验证集只 transform

# ---- 模型 ----
clf = LogisticRegression(max_iter=1000)
clf.fit(X_tr, y_tr)
p = clf.predict_proba(X_va)[:, 1]        # 正类概率

# ---- 指标 ----
roc_auc_score(y_va, p)                   # AUC
log_loss(y_va, p)                        # LogLoss
mean_squared_error(y_va, pred, squared=False)  # RMSE
r2_score(y_va, pred)                     # R²
```

**关键纪律**：所有 `fit`（标准化、填充、编码）只能在训练折上做，验证/测试只 `transform`。否则泄漏。

---

## B.2 XGBoost 速查

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=2000,          # 树的数量
    learning_rate=0.05,         # 学习率
    max_depth=6,                # 树深度
    subsample=0.8,              # 行采样
    colsample_bytree=0.8,       # 列采样
    eval_metric="auc",
    early_stopping_rounds=100,  # 早停
    random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
p = model.predict_proba(X_va)[:, 1]
```

**类别不平衡**：`scale_pos_weight = 负类数 / 正类数`。

---

## B.3 LightGBM 速查

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,              # 叶子数（控制复杂度）
    min_child_samples=20,       # 叶子最小样本
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
p = model.predict_proba(X_va)[:, 1]
```

**LightGBM vs XGBoost**：
- LightGBM 更快（直方图算法）、内存更省；
- XGBoost 在某些数据上更稳；
- 两者都强，集成时通常都放。

---

## B.4 三库对比速查

| 需求 | 工具 |
|------|------|
| 通用预处理/切分/指标 | scikit-learn |
| 表格强基线（单模型） | LightGBM / XGBoost |
| 类别不平衡 | `scale_pos_weight`（XGB）/ `class_weight`（LGBM） |
| 早停 | `early_stopping_rounds` / `early_stopping` |
| 特征重要性 | `model.feature_importances_` |

下一附录是 AutoGluon 速查。