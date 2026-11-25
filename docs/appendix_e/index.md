# 附录E：表格基础模型速查（TabPFN 等）

> 表格基础模型（Tabular Foundation Model, TFM）是近两年兴起的新范式：像大语言模型一样，在海量表格上预训练，推理时无需训练/微调即可预测。中小规模数据上表现极强。

---

## E.1 TabPFN

```python
from tabpfn import TabPFNClassifier, TabPFNRegressor

# 分类
clf = TabPFNClassifier(device="cuda")
clf.fit(X_train, y_train)             # 注意：它内部不训练，只是"记住"数据
p = clf.predict_proba(X_test)[:, 1]

# 回归
reg = TabPFNRegressor(device="cuda")
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
```

**适用场景**：
- 样本数 < 1 万、特征 < 500（超出会慢/效果下降）；
- 想要“零调参”的强基线；
- 作为集成多样性来源（和树模型互补）。

**注意**：TabPFN 对数据规模敏感，大数据上不如 GBDT；它也不是“万能”，中小表格才是主场。

---

## E.2 其他表格基础模型

| 模型 | 特点 | 来源 |
|------|------|------|
| TabPFN | 零样本/少样本，中小数据极强 | Prior Labs |
| TabICL | 上下文学习，大数据可用 | 2025 |
| Mitra | 混合合成先验增强 | 2025 |
| Nori / TabDPT-Turbo | AutoGluon `extreme` 预设内置 | TabArena |

这些模型大多可以通过 AutoGluon 的 `presets="extreme"` 或 `noncommercial` 自动调用，无需单独写代码。

---

## E.3 在 VibeKaggle 中的定位

- **第一个合规工件的快速来源**：TabPFN 零调参，几秒出结果；
- **多样性来源**：和 AutoGluon 的树模型集成，往往涨分；
- **不替代 GBDT**：大数据、高基数类别，GBDT 仍是主力。

下一附录是 Kaggle MCP 配置与安全实践。