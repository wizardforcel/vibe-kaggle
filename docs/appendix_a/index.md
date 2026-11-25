# 附录A：NumPy / pandas / matplotlib 快速参考（精简合并版）

> 本附录是全书代码里反复出现的三个库的速查表。不需要背，遇到忘了的语法来翻一翻即可。

---

## A.1 NumPy 速查

NumPy 是 Python 做数值计算的基础库，核心对象是 `ndarray`（多维数组）。

```python
import numpy as np

# ---- 创建数组 ----
a = np.array([1, 2, 3])              # 从列表创建
b = np.zeros((3, 4))                 # 全 0 矩阵
c = np.ones(5)                       # 全 1 向量
d = np.arange(0, 10, 2)              # 0,2,4,6,8（等差）
e = np.linspace(0, 1, 5)             # 0,0.25,0.5,0.75,1（等间隔）

# ---- 形状操作 ----
a.reshape(3, 1)                      # 改成 3 行 1 列
a.ravel()                            # 拉平成一维
np.concatenate([a, b], axis=0)       # 拼接
np.stack([a, b], axis=0)             # 堆叠成新维度

# ---- 常用统计 ----
a.mean(), a.std(), a.min(), a.max()  # 均值、标准差、最小、最大
np.percentile(a, [25, 50, 75])       # 分位数
np.median(a)                         # 中位数

# ---- 布尔索引（筛选） ----
a[a > 1]                             # 取大于 1 的元素
np.where(a > 1, a, 0)                # 大于 1 保留，否则置 0

# ---- 线性代数 ----
M = np.random.rand(3, 3)
M.T                                  # 转置
np.dot(M, M.T)                       # 矩阵乘法
np.linalg.inv(M)                     # 求逆
```

**VibeKaggle 常用模式**：
- OOF 数组：`oof = np.zeros(len(train), dtype=np.float32)`，按折填充；
- 保存/加载：`np.save("x.npy", arr)` / `arr = np.load("x.npy")`；
- 拼接多个模型预测：`np.column_stack([oof1, oof2])`。

---

## A.2 pandas 速查

pandas 是表格数据处理的核心库，核心对象是 `DataFrame`（带行列标签的表）。

```python
import pandas as pd

# ---- 读写 ----
df = pd.read_csv("train.csv")                # 读 CSV
df.to_csv("out.csv", index=False)            # 写 CSV（不写行号）

# ---- 查看 ----
df.head()                                    # 前 5 行
df.info()                                    # 列类型、缺失情况
df.describe()                                # 数值列统计
df.shape                                     # (行数, 列数)

# ---- 选择列/行 ----
df["col"]                                    # 单列
df[["a", "b"]]                               # 多列
df.loc[0:10, "a"]                            # 按标签选
df.iloc[0:10, 0]                             # 按位置选

# ---- 缺失值 ----
df.isna().sum()                              # 每列缺失数
df.fillna(df.median())                       # 用中位数填充
df.dropna()                                  # 删缺失行

# ---- 分组聚合（特征工程核心） ----
df.groupby("cat")["num"].mean()              # 按类别求均值
df.groupby("cat")["num"].agg(["mean", "std", "count"])  # 多个统计量
df.groupby("cat")["num"].transform("mean")   # 广播回原形状（做特征常用）

# ---- 时间序列 ----
df["date"] = pd.to_datetime(df["date"])      # 转日期
df["date"].dt.dayofweek                      # 星期几
df["date"].dt.month                          # 月份
df.sort_values("date").groupby("id")["y"].shift(1)  # 滞后（防泄漏）
```

**VibeKaggle 常用模式**：
- 读三份文件：`train/test/sample_submission`；
- 分组聚合做特征：`groupby(...).transform(...)`；
- 时序滞后：`groupby(id).shift(k)`，滚动 `groupby(id).shift(1).rolling(w)`。

---

## A.3 matplotlib 速查

matplotlib 是画图的基础库。

```python
import matplotlib.pyplot as plt

# ---- 折线图 ----
plt.plot(x, y, label="line")
plt.legend(); plt.show()

# ---- 柱状图 ----
plt.bar(categories, values)

# ---- 直方图 ----
plt.hist(data, bins=30)

# ---- 散点图 ----
plt.scatter(x, y, c=colors)

# ---- 子图 ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(img)

# ---- 保存（书里都用这个） ----
plt.savefig("out.png", dpi=120, bbox_inches="tight")
plt.close()   # 画完关掉，释放内存
```

**VibeKaggle 常用模式**：
- EDA 出图：目标分布、箱线图、相关性热力图（`plt.imshow(corr)`）；
- 接触表：`subplots` 拼多张图供人审查；
- 匹配可视化：两图并排 + 连线。

---

## 小结

这三个库是全书代码的“字母表”。不需要一次记住，用到时回来查。下一附录是机器学习库的速查。