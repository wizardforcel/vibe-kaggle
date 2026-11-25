# 《VibeKaggle》

> **副标题**：用大模型打 Kaggle 的全新人机协作方式
>
> **本书定位**：面向会一点 Python、未参加过 Kaggle 竞赛的读者。它不教你“亲手做完每一个步骤”，而是教你**设计并运行一条人机协作的竞赛工作流**：让自动化工具和 LLM Agent 承担可重复的执行与枚举，你把有限的时间花在验证设计、方向判断和工件审查上。
>
> **VibeKaggle 的工作定义（贯穿全书）** ：以**工件驱动**为物理基础，以**验证优先**为判断纪律，以**人定方向、Agent 执行**为分工原则，以**提交预算**为安全边界，在 LLM Agent 的吞吐能力之上，重新组织 Kaggle 竞赛的每一个环节。


## 前言：VibeKaggle 不是“用 AI 写代码”

Google 与 Kaggle 在 2026 年发布的《The New SDLC With Vibe Coding》给出了一个关键判断：随意提示和真正的工程之间，分水岭是**验证**。这份白皮书进一步提出了一个对全书具有组织意义的分解：**Agent = 模型 + Harness，比例大约是 10/90**。模型只占 10%，harness——指令、工具、MCP 服务器、沙箱、编排、护栏、可观测性——占 90%。大多数 Agent 失败，是**配置失败**，不是模型能力失败。这意味着 VibeKaggle 的核心技能不是“写更好的提示”，而是**设计更好的 harness**：固定验证协议、规范工件命名、限制提交预算、约束工具集。

NVIDIA 的 Chris Deotte 在 2026 年 3 月的 Kaggle Playground 客户流失预测竞赛中，用 GPT-5.4 Pro、Gemini 3.1 Pro 和 Claude Opus 4.6 三个 LLM Agent 生成了超过 60 万行代码，运行了 850 个实验，构建了一个由 150 个模型组成的四级 stack 并获得第一名。他的工作流中贯穿一条纪律：每个实验的 OOF 和测试预测都必须保存为可组合的工件文件。没有这条纪律，850 个实验只会制造不可复现的混乱。

与此同时，Agent 的能力边界不是固定的。MLE-Bench 的基线显示纯自动 Agent 的奖牌率约为 16.9%，但在 harness 设计良好的条件下，先进系统已经能把这一数字推到远高于此的水平。**边界的位置取决于 harness 设计，而非模型选型**。人不可让渡的角色，是在 Agent 的高吞吐量之上做**判断**——判断哪个方向值得深入、验证方案是否系统性偏差、工件是否可信、是否值得消耗一次提交。

这本书的组织方式因此与第一版根本不同：**工件规范不是附录，而是第 2 章的基础协议；自动基线不是可选项，而是每一个代表比赛章的第一个动作；Agent 协作不是策略章节的话题，而是贯穿全书的操作模板。**


## 第1章 VibeKaggle 的工作流全景：从第一天起就在协作中工作

**1.1 Kaggle 的五分钟导览**
比赛页面、数据页面、讨论区、排行榜、Notebooks 五大区域的位置与作用；区分“比赛”与“数据集”。【Mermaid：比赛页面导览图】

**1.2 选择一场“自动化友好”的入门比赛**
指标明确、有 `sample_submission`、讨论区有活跃的公开 notebook 可供 Agent 参考。Playground Series 是理想的第一站：2025—2026 年的 Playground 覆盖了从贴纸销售预测到客户流失、从道路事故风险到 F1 进站预测的多种表格任务，数据规模可控，评估指标清晰，且获胜方案公开可查。

**1.3 三份核心文件与提交格式**
`train.csv` / `test.csv` / `sample_submission.csv` 各自的内容与约束；Agent 生成提交文件时的**格式自检清单**。

**1.4 VibeKaggle 的第一个动作：建立工件目录，而不是先写模型**
在跑任何模型之前，创建 `artifacts/oof/`、`artifacts/test_preds/`、`artifacts/logs/`，确定命名规范：`train_oof_[MODEL]_[VERSION].npy` 和 `test_preds_[MODEL]_[VERSION].npy`。第一个基线——哪怕是一个常数预测——也必须产出合规的 OOF 和 test prediction 文件。没有工件，Agent 的高吞吐量只会制造混乱。【Mermaid：工件目录结构与数据流向图】

**1.5 三种提交路径及其安全边界**
手动网页上传 / Kaggle CLI / MCP 驱动的提交。核心原则：Agent 可以让提交“零摩擦”，但**零摩擦是风险不是优势**。提交预算和预览-确认机制是工作流的护栏。

**1.6 排行榜解读**
Public LB 在 Agent 高频实验下噪声被放大；本地 CV 与 LB 的差异在 Agent 时代的解读方式。

**1.7 规则红线（精简版）**
外部数据、预训练模型、自动化行为的披露边界。

**1.8 小结**
本章结束时，读者应该拥有一个**带工件目录的空工作区**，以及一个已跑通的常数基线工件。


## 第2章 VibeKaggle 的操作系统：工件、验证、Agent、预算

**2.1 工件规范：VibeKaggle 的物理基础**
命名约定、目录结构、每个工件必须附带的最小元数据（CV 分数、验证折索引、一行改动描述）。没有工件规范，“人看 OOF 做判断”是空话，“Agent 自动做 hill climbing”无从谈起，“验证污染”也无从检查。【Mermaid：工件生命周期图——从基线到集成】

**2.2 验证优先：不可委托的判断纪律**
验证方案的设计（折数、分层、分组、时间切分）必须由人完成，Agent 只负责执行。**验证污染**：当 Agent 被允许“尝试多种 CV 方案并选最好的那个”时，验证本身就成了搜索空间的一部分。人必须**固定验证协议**。【Mermaid：验证污染的形成机制图】

**2.3 LLM Agent 的能力边界：执行和枚举，而非判断**
Agent 擅长：EDA 代码生成、baseline pipeline 编写、批量特征变换实验、超参数扫描、实验结果汇总、stacking 组合的候选生成。Agent 不擅长：判断哪个方向值得深入、识别验证方案中的系统性偏差、在语义层面判断特征是否合理、决定是否提交。人向 Agent 发出的指令应该是**带约束的请求**，而不是“帮我建个模型”。

**2.4 AutoML 与表格基础模型：强基线的快速来源**
AutoGluon 的多层 stacking 能力使其成为表格竞赛中建立“不费力竞争基线”的首选；TabPFN 等表格基础模型在中小规模数据上提供零调参的替代基线。AutoML/基础模型产出**第一个有 CV 分数的合规工件**，人据此判断“当前数据上不费力能做到多好”。

**2.5 MCP：让 Agent 持续操作平台，安全是内置的**
MCP 的核心价值：让下载→EDA→训练→工件落盘→提交预演成为 Agent 可持续运行的一条流水线。安全机制是这条流水线的护栏：预览-确认双调用、每日提交预算、不可逆操作默认关闭。

**2.6 本章的验收任务**
在同一个表格数据集上，用 AutoGluon 跑出一个合规基线工件；用 LLM Agent 生成一版 EDA 代码并执行，人工标注可靠与需核实的结论；配置一次 MCP 提交预演（dry-run）。


## 第一部分 表格数据：VibeKaggle 的主场

> 表格数据是 AutoML 和 Agent 成熟度最高、收益最直接的模态。本部分的每一章都从**自动基线产出合规工件**开始，然后展示人在哪里介入、Agent 在哪里执行、工件如何支撑判断。


### 第3章 表格数据的 Vibe 工作流：从自动基线到人机迭代

**3.1 第一个动作：自动基线 + 工件落盘**
用 AutoGluon 在原始数据上直接 `fit`，**验收标准不是分数，而是产出合规的 OOF 和 test prediction 文件**；同时跑一个 TabPFN 基线作为多样性来源。【Python 代码示例：AutoGluon 拟合 + 工件保存】【Mermaid：自动基线的工作流位置图】

**3.2 Agent 执行、人审查的 EDA**
Agent 生成 EDA 代码，人审查**哪些结论是数据支持的、哪些是幻觉或过度概括**。人的 EDA 重点放在 Agent 容易忽略的地方：缺失值的语义、目标偏态对验证空间的影响。

**3.3 特征构造：人类判断力最该介入的地方**
AutoGluon 能自动生成交互特征，但**领域驱动的组合特征**仍然是人工优势。人提出特征假设，Agent 在同一 CV 协议下生成对比工件，人根据 CV 和工件质量决定是否保留。

**3.4 验证协议：人设定，Agent 执行**
交叉验证的折数、分层方式、时间切分由人根据数据性质决定；Agent 执行 CV，但**验证方案的设计和泄漏检查是人不可让渡的职责**。

**3.5 从工件到集成：Agent 做搜索，人做判断**
AutoGluon 已做多层 stacking；人工 stack 的价值在于引入 **AutoGluon 之外的模型家族**。用 OOF 工件做 hill climbing 和 meta-model stacking 的基本流程；**人的判断**：哪些模型进入集成、权重是否稳定、复杂度是否值得。

**3.6 表格 Vibe 循环的操作模板（全书复用）**
人给出带工件约束的指令 → Agent 执行并产出工件 + 日志 → 人审查 CV、OOF 质量、泄漏风险 → 人决定保留/丢弃/修改方向 → Agent 执行下一批。【Mermaid：五步循环图】

**3.7 小结**


### 第4章 表格比赛详解：Playground S6E3 客户流失预测（2026，NVIDIA LLM Agent 夺冠）

> 这场比赛是 VibeKaggle 的“标准示范”：Agent 的高吞吐量、工件规范和人的验证纪律在同一场比赛中全部可见。

**4.1 比赛与数据**
电信客户流失二分类，AUC 指标。Playground Series S6E3，2026 年 3 月。

**4.2 自动基线 + 合规工件**
AutoGluon 或 LightGBM 快速基线，产出 OOF 和测试预测工件。人检查概率分布和 AUC 的置信区间。

**4.3 Agent 批量实验与人的验证门控**
NVIDIA 的方案中，三个 LLM Agent 在固定验证协议下运行了 850 个实验，从 850 个模型中选出 150 个构建四级 stack。本章的核心教学点不是“复制这个规模”，而是**理解门控规则**：Agent 被允许在固定 CV 折上枚举模型和特征组合，但**验证协议本身在整个过程中保持不变**；人检查实验日志中是否有 Agent 尝试“换一个 CV 方案来获得更好分数”的迹象。

**4.4 工件审查：从 850 到 150 的选择**
人审查的是：OOF 相关性结构、模型多样性、集成后的 CV 稳定性。Agent 做候选生成和初步筛选，但**进入最终 stack 的模型子集由人确认**。

**4.5 提交纪律**
提交预算的消耗点、概率校准检查、格式核对。【Python 代码示例：提交文件生成 + 断言检查】

**4.6 小结：这场比赛展示了 VibeKaggle 的完整循环——Agent 的吞吐、工件的可组合性、人的验证门控和选择判断**


### 第5章 表格比赛详解：Playground S5E10 道路事故风险预测（2025，AutoGluon 作为起点）

**5.1 比赛与数据**
回归任务，RMSE 指标，2025 年 Playground。

**5.2 自动基线工件**
AutoGluon 适配器快速产出基线工件；8th place 方案明确使用 AutoGluon 作为基线模型之一，并在原型阶段用 AutoGluon 探索可行性。

**5.3 Agent 枚举与人的模型选择**
Agent 批量生成 XGBoost、TabM、LightGBM、神经网络等模型的工件；人审查的是：哪些模型的 OOF 提供了**互补信号**，而不是单纯看单模型分数。8th place 方案从约 40 列 OOF 中通过 Greedy NNLS 和 LassoCV 筛选到 4—6 列，这个**筛选逻辑**是人判断的产物。

**5.4 集成的人机分工**
Agent 执行 Ridge/MetaNN/残差 stacking 的候选实验；人判断 meta-learner 的复杂度和 CV 稳定性是否值得。

**5.5 提交**
【Python 代码示例：提交文件生成 + 核对】

**5.6 小结：AutoGluon 建立基线，Agent 扩大候选池，人做互补性判断**


### 第6章 表格比赛详解：Playground S6E1 学生考试成绩预测（2025—2026，特征工程的深度挖掘）

**6.1 比赛与数据**
连续目标 `exam_score`，RMSE 指标。

**6.2 自动基线工件**
AutoGluon / LightGBM 基线工件。

**6.3 Agent 枚举特征变体，人判断哪些有语义合理性**
1st place 方案包含：模数字段分解、谐波周期三角函数特征、带经验贝叶斯平滑的 OOF 多聚合目标编码、PyTabKit RealMLP + GBDT 模型动物园、Centered Isotonic Regression 校准 + Ridge stacking。Agent 可以批量生成这些变体的工件，但**“模数字段分解是否合理”“谐波周期选哪些周期”** 需要人根据数据生成机制做出判断。

**6.4 验证纪律与过拟合门控**
人设定固定的 5 折 CV；Agent 在固定折上迭代特征和模型。人检查 Agent 是否在无意中使用了未来信息（例如在目标编码中使用了验证折的目标值）。

**6.5 集成与提交**
【Python 代码示例：多工件集成 + 提交核对】

**6.6 小结：特征工程的深度依然是人可以胜过纯自动化的领域，但 Agent 让枚举变体的成本几乎归零**


### 第7章 表格比赛详解：NeurIPS Open Polymer Prediction 2025（2025，科学表格数据的验证特殊性）

**7.1 比赛与数据**
NeurIPS 2025 的聚合物性质预测，$50,000 奖金竞赛，回归/多任务表格任务。

**7.2 自动基线工件与领域约束**
AutoGluon 基线工件；人检查预测值是否落在物理合理的范围内。科学数据的验证不能只看 CV——**领域约束是人的判断**。

**7.3 Agent 批量实验，人审核物理合理性**
Agent 枚举模型和特征变体；人审查工件中的预测分布和异常值，判断是否存在“分数好但物理上不可信”的模型。

**7.4 集成与提交**
【Python 代码示例：提交文件生成 + 物理约束断言】

**7.5 小结：科学表格竞赛中，人的判断不仅是验证纪律，还包括领域合理性的守门**


## 第二部分 时序数据：在 Agent 吞吐与时间纪律之间

### 第8章 时序数据的 Vibe 工作流：自动化枚举与人工时间纪律

**8.1 时序自动基线 + 工件落盘**
AutoGluon-TimeSeries 快速基线；基础模型（Chronos/TimesFM）的零样本预测作为对照工件。

**8.2 Agent 执行、人审查的特征工程**
日期拆分交给 Agent；滞后阶数和滚动窗口的**候选枚举**交给 Agent，但**哪些阶数有周期含义**由人根据 EDA 判断。

**8.3 时间验证：不可委托的纪律**
人设定时间切分方案，Agent 执行；Agent 的任何“随机划分”建议应被默认拒绝；人检查工件中的验证折索引是否符合时间顺序。

**8.4 基础模型与树模型的工件级融合**
基础模型工件和树模型工件并列；人判断融合方式（简单加权、残差建模、将基础模型预测作为特征送入树模型）。

**8.5 小结**


### 第9章 时序比赛详解：Playground S5E1 贴纸销售预测（2025，时间纪律的入门标准）

**9.1 比赛与数据**
预测不同国家、商店和产品的贴纸销售额，MAPE 指标，2025 年 Playground Series S5E1。

**9.2 自动基线工件**
AutoGluon-TimeSeries / LightGBM 基线工件；人检查不同国家/商店的误差模式。

**9.3 人设定 Group K-Fold 时间纪律**
按商店和产品分组做时间切分，Agent 执行；**人检查工件中的折索引是否严格按时间顺序**。Top 15% 的公开方案使用 Group K-Fold 验证，说明时间/分组纪律从入门级比赛就是核心。

**9.4 Agent 枚举滞后与滚动特征，人判断周期含义**
Agent 批量生成滞后 1/7/14/28 天和滚动窗口的工件；人根据零售数据的周周期和月度效应判断哪些窗口有业务含义。

**9.5 集成与提交**
【Python 代码示例：多工件融合 + 提交核对】

**9.6 小结：入门级时序比赛的首要教学点不是模型复杂度，而是时间验证纪律**


### 第10章 时序比赛详解：Playground S5E10 道路事故风险预测（2025，Agent 枚举与人做互补性选择）

> 这场比赛同时具有表格和时序特征，放在时序部分侧重**时间切分下的 Agent 实验管理**。

**10.1 比赛与数据**
回归任务，RMSE；数据包含时间相关的道路/环境特征。

**10.2 自动基线在时间切分下的工件**
AutoGluon 基线在时间切分下的 CV；与随机 CV 的分数差异由人记录和解读。

**10.3 Agent 枚举模型，人选择互补子集**
Agent 生成 XGBoost、TabM、LightGBM、NN 的工件；人通过 OOF 相关性矩阵和 Greedy NNLS 判断哪些模型值得进入集成，而不是让 Agent 自动做 hill climbing 后直接提交。

**10.4 提交**
【Python 代码示例：提交文件生成 + 时间切分验证记录】

**10.5 小结：Agent 扩大模型候选池，人的判断集中在“互补性”而非“单模型最高分”**


### 第11章 时序比赛详解：Jane Street Real-Time Market Data Forecasting（2024—2025，非平稳时序的验证极限）

**11.1 比赛与数据**
金融高频数据，加权零均值 R²（R2 for `responder_6`），2024 年 10 月—2025 年 7 月。数据的肥尾、非平稳和突变特征使标准 CV 假设几乎全部失效。

**11.2 自动基线的局限**
AutoGluon-TimeSeries / 树模型基线在非平稳金融数据上的表现往往不稳定；人必须**先判断自动基线是否可信**，再决定是否投入 Agent 实验资源。

**11.3 时间验证的极端纪律**
官方评估 API 强制模型不能“向前看”。人设定严格的时序切分；Agent 的任何交叉验证建议都必须经过人审核。**这是全书最强调“Agent 不应自主选择验证方案”的一章**。

**11.4 Agent 枚举特征与模型，人审核非平稳性**
Agent 可以批量尝试不同特征组合和正则化强度；人审查工件在不同时间段上的稳定性，而不是只看全局 CV。

**11.5 提交**
【Python 代码示例：提交文件生成 + 稳定性断言】

**11.6 小结：当验证本身极其困难时，人的判断不是“选一个分数”，而是“判断这个分数是否可信”**


## 第三部分 图像数据：基础模型时代的人机分工

### 第12章 图像数据的 Vibe 工作流

**12.1 默认起点：预训练骨干 + 微调，工件先行**
DINOv3、ViT 等预训练骨干的微调是主路径；第一个微调基线必须产出验证集概率（OOF）和测试集预测工件。

**12.2 Agent 管理实验矩阵，人审查增强策略**
Agent 批量运行不同骨干、分辨率、增强组合的微调实验，汇总工件和验证分数；**增强策略必须与验证方案一起评估**，更激进的增强是否匹配数据特性由人判断。

**12.3 类别不平衡与指标特殊性**
医学图像中，Agent 的自动类别加权可能忽视临床意义上的差异；人根据指标判断加权策略。

**12.4 提交前的工件级检查（人执行）**
图像提交的错误代价高：通道顺序、归一化方式、ID 映射、TTA 后的概率校准。人对照工件和 `sample_submission` 逐项核对。

**12.5 小结**


### 第13章 图像比赛详解：CSIRO Image2Biomass Prediction（2025—2026，DINOv3 多模态融合）

**13.1 比赛与数据**
从牧场图像 + 表格传感器数据（NDVI、高度）预测生物量，2025 年 10 月—2026 年 1 月。

**13.2 自动基线工件**
DINOv3 ViT 骨干 + 轻量回归头的基线工件。4th place 方案使用 `vit_huge_plus_patch16_dinov3.lvd1689m`，前 50% 层冻结。

**13.3 Agent 实验矩阵，人判断数据清洗**
4th place 方案的关键增益之一来自**人工数据清洗**：识别并裁剪图像中的“纸板”背景噪声，带来 +0.01 的 LB 提升。Agent 不会自动发现这种数据质量问题；人通过视觉检查提出清洗规则，Agent 执行并产出对比工件。

**13.4 多模态融合的人机分工**
Agent 枚举图像特征与表格特征的融合方式；人根据领域含义判断融合策略的合理性。4th place 方案使用辅助任务训练和多模态融合，这些设计选择需要人的领域判断。

**13.5 集成与提交**
【Python 代码示例：多骨干工件集成 + 提交核对】

**13.6 小结：基础模型提供强基线，但数据清洗和融合设计仍然是人的判断领域**


### 第14章 图像比赛详解：RSNA Knee Abnormality Detection（2026，医学影像 + LLM Agent 监督）

**14.1 比赛与数据**
膝关节 MRI 多标签异常检测，12 类临床异常，2026 年 RSNA 挑战赛。

**14.2 自动基线工件**
DINOv3 / DINOv2 骨干微调基线；人检查各类别的预测概率分布和混淆矩阵。

**14.3 LLM Agent 作为监督来源（本章的核心新内容）**
2026 年 RSNA Knee 的 1st place 方案使用了 **frontier LLM agents 来获取训练集的不同类型监督信号**。这代表了 VibeKaggle 的一个前沿形态：Agent 不只是执行代码，还可以**生成辅助监督**。人的判断在于：哪些 LLM 生成的监督信号在临床上是可信的，哪些可能引入系统性偏差。

**14.4 Agent 枚举骨干与增强，人审核临床合理性**
Agent 批量运行 DINOv3、DINOv2 等骨干的微调工件；人根据医学图像的语义判断增强策略是否可能改变病变特征。

**14.5 集成与提交**
【Python 代码示例：多骨干集成 + 提交核对】

**14.6 小结：医学影像中 Agent 可以参与监督生成，但临床合理性的守门人仍然是人**


### 第15章 图像比赛详解：Image Matching Challenge 2025（2025，几何视觉中的 Agent 实验管理）

**15.1 比赛与数据**
图像匹配任务，评估特征匹配和几何验证的准确性，2025 年 Kaggle 竞赛。

**15.2 自动基线工件**
预训练匹配模型（如 SuperPoint/LightGlue 类方法）的基线工件；人检查匹配成功率在不同场景下的分布。

**15.3 Agent 管理匹配管线的实验矩阵**
图像匹配管线包含特征提取、匹配、几何验证、RANSAC 等多个阶段。Agent 可以批量管理不同组合的实验工件；人判断**哪个阶段的改动带来了真实增益**，而不是被整体分数的波动误导。

**15.4 提交前的可视化审查（人的不可替代性）**
图像匹配的错误往往在数值指标上不明显，但在可视化中一目了然。人必须**亲自查看匹配可视化工件**，而不是仅依赖 Agent 汇总的分数。

**15.5 提交**
【Python 代码示例：提交文件生成 + 匹配可视化检查清单】

**15.6 小结：在几何视觉任务中，人的视觉审查仍然是 Agent 汇总指标无法替代的判断环节**


## 第四部分 文本数据：LLM 原生赛道的双重角色

### 第16章 文本数据的 Vibe 工作流

**16.1 现代起点：Transformer 微调 + 工件落盘**
DeBERTa/RoBERTa/开源 LLM 微调是默认强基线；RAG 在需要外部知识的任务中成为常见组件。第一个微调基线必须产出验证集概率和测试预测工件。

**16.2 Agent 做清洗，人定义“保留什么”**
否定词、转折词、反讽标记是否需要特殊保留，由人根据任务语义判断；Agent 不应在未授权的情况下删除这些信号。

**16.3 特征工程的 Vibe 分化**
Transformer 嵌入作为主信号；人工设计的相似度/规则特征作为多样性来源。Agent 批量生成变体，人判断语义合理性。

**16.4 LLM 作为竞赛模型的使用纪律**
人设定 API 预算和推理成本约束；Agent 不应在未授权的情况下消耗大量额度。

**16.5 小结**


### 第17章 文本比赛详解：Jigsaw — Agile Community Rules Classification（2025，LLM 微调与验证策略的判断）

**17.1 比赛与数据**
Reddit 评论是否违反特定社区规则，二分类，列平均 AUC，2025 年 Jigsaw 竞赛。

**17.2 Transformer 基线工件**
Qwen/Llama/Phi 小模型微调基线；Agent 管理多模型工件。

**17.3 验证策略的不可委托性（本章核心）**
1st place 方案明确选择**信任 Public LB 作为验证**，理由包括 LB 切分随机、测试数据量足够大以降低方差。这个判断是**人做出的验证方案选择**——Agent 不应自主决定“用 LB 还是 CV”。在文本竞赛中，训练/测试分布往往不完全一致，验证策略的选择本身就是一个需要人判断的问题。

**17.4 Agent 执行微调实验，人审核损失掩码与数据增强**
1st place 方案在微调时**手动排除 “Yes”/“No” 之外的 token 的损失**，并利用 test.csv 中的正负例进行上采样。这些操作由人设计，Agent 批量执行不同配置的工件。

**17.5 集成与提交**
【Python 代码示例：多模型工件集成 + 提交核对】

**17.6 小结：当验证策略本身成为判断对象时，人的角色是“选择用什么标准来判断”，而不是“让 Agent 选一个分数最高的标准”**


### 第18章 文本比赛详解：LLM Prompt Recovery（2024，生成式任务中的混合策略判断）

**18.1 比赛与数据**
从改写后的文本恢复原始 prompt，评估指标与语义相似度相关，2024 年 Google 竞赛。

**18.2 自动基线工件**
Mistral/Gemma 等 LLM 微调的基线工件。

**18.3 Agent 枚举混合策略，人判断“均值 prompt + 模型预测”的组合**
3rd place 方案采用 **“mean prompt + 标签 + 完整 prompt（如果通过门控）”** 的混合策略。Agent 可以批量生成不同组合的工件；人判断门控条件是否在语义上合理——例如，什么情况下模型预测应该被“mean prompt”覆盖。

**18.4 生成式任务的验证特殊性**
生成式任务的评估指标可能无法完全捕捉语义质量；人需要通过人工抽检工件来判断 Agent 优化的方向是否偏离。

**18.5 提交**
【Python 代码示例：混合策略工件生成 + 提交核对】

**18.6 小结：生成式文本任务中，“什么是好输出”的判断不能完全交给指标，更不能完全交给 Agent**


### 第19章 文本比赛详解：Make Data Count — Finding Data References（2025，检索 + 分类的混合流程）

**19.1 比赛与数据**
从科学论文全文中识别数据引用并分类为主要/次要引用，2025 年竞赛。

**19.2 自动基线：检索 + 分类两阶段工件**
1st place 方案使用论文和数据集元数据训练 CatBoost；4th place 使用 EuropePMC 论文生成合成标签用于 Qwen2.5 预训练。基线工件包括检索结果和分类概率。

**19.3 Agent 管理两阶段实验，人判断稀有类别的处理**
Agent 可以批量尝试不同的检索策略和分类器组合；人判断**检索阶段的假阳性/假阴性对最终分数的影响**，以及次要引用这类稀有类别的处理方式。

**19.4 提交**
【Python 代码示例：两阶段工件合并 + 提交核对】

**19.5 小结：多阶段文本流程中，人的判断集中在“阶段之间的误差传播”上，而不是单个阶段的分数**


## 第20章 验证与评估：Vibe 时代更不可让渡的判断

**20.1 指标选择的不可委托性**
RMSE/MAE/RMSPE、AUC/LogLoss 的选择是人的判断，不是 Agent 的搜索空间。

**20.2 交叉验证方案的设计：Agent 执行，人设计并审核**
折数、分层、分组、时间切分的方案由人设计；Agent 执行并产出工件，人检查工件中的验证折索引。

**20.3 验证污染：Agent 时代最隐蔽的过拟合**
当 Agent 被允许尝试多种 CV 方案并选最好的那个，验证本身就成了被优化的对象。人必须**固定验证协议**。

**20.4 目标泄漏检查清单（升级版）**
增加“Agent 生成的特征是否无意中包含未来信息”和“工件的验证折索引是否符合时间/分组约束”。

**20.5 本地 CV 与 LB 差异在 Agent 高频提交下的解读**
提交预算本身就是一种强制纪律：有限的提交次数迫使 Agent 不能在 LB 上做穷举搜索。

**20.6 小结**


## 第21章 调优与集成：Agent 搜索，人选择

**21.1 超参数搜索 largely 委托给 Agent/AutoML**
人设定搜索空间和计算预算；Agent 执行并产出工件。

**21.2 集成的“选择”不是搜索问题**
哪些模型进入集成、用什么 meta-learner、权重是否稳定，需要人查看 OOF 工件的质量和相关性结构。NVIDIA 的冠军方案从 850 个模型中选出 150 个做四级 stacking，**选择过程**包含人对 OOF 质量和多样性的判断。

**21.3 AutoGluon 的多层 stacking vs 人工定制的 stacking**
AutoGluon 的 stacking 是自动化的强基线；人工 stack 的价值在于引入 AutoGluon 之外的模型家族。

**21.4 Agent 驱动的集成实验管理**
工件规范让 Agent 可以自动做 hill climbing 的候选生成；人审核最终集成的 CV 稳定性和复杂度代价。

**21.5 集成收益的边际递减与过拟合风险**
【Mermaid：集成复杂度与 CV 收益的边际曲线图】

**21.6 小结**


## 第22章 VibeKaggle 的策略与心态

**22.1 选择比赛的标准**
增加“自动化工具在该模态上的成熟度”维度：表格比赛的 Agent 杠杆最高，图像/时序的杠杆较低但仍在快速演进。

**22.2 先让 Agent 跑自动基线，但不要在第一小时就提交**
先看 CV 和工件质量，再决定是否消耗提交预算。

**22.3 提交预算作为工作流的内置约束**
Agent 让提交“零摩擦”，但零摩擦是风险。人必须保留**提交决策权和预算控制权**。

**22.4 团队协作的新形态：人类队友 + Agent 队友**
人负责方向和验证，Agent 负责实验吞吐和工件生产。

**22.5 从“我会做 Kaggle”到“我会指挥系统做 Kaggle”**
竞赛履历的价值正在从手工能力转向**工作流设计能力和判断质量**。

**22.6 小结**


## 第23章 从 VibeKaggle 到实际项目

**23.1 VibeKaggle 工作流与生产 ML 流水线的异同**
工件规范、验证纪律和提交控制在生产中同样是核心问题。

**23.2 把 Agent 辅助的竞赛代码变成可维护项目**
实验日志、工件管理、配置管理、可复现性检查。

**23.3 模型部署基础（精简保留）**
【Python 代码示例：FastAPI 预测接口简例】

**23.4 继续学习的资源**
MLE-Bench、Agentic Kaggle Skill、Kaggle 社区中 Agent 辅助的 writeup。

**23.5 全书知识点回顾**


## 附录A：NumPy / pandas / matplotlib 快速参考（精简合并版）

> 定位：把三件套最常用的 20% 操作集中在一处，供全书代码片段回查。不追求 API 全量，只保留"写竞赛代码真的会用到"的部分。

**A.1 NumPy 数组创建与形状变换**
`np.array` / `np.zeros` / `np.arange` / `np.linspace` 的创建方式；`reshape` / `ravel` / `transpose` / `concatenate` / `stack` 的形状操作；形状不匹配时 `ValueError` 的常见原因。

**A.2 NumPy 索引、切片与广播**
基本切片与视图（view）vs 拷贝（copy）的区别；布尔掩码索引；`np.where` 三目运算；广播规则（broadcast）与"为什么 (n,1) 和 (n,) 能相加"。

**A.3 NumPy 统计与聚合**
`sum` / `mean` / `std` / `min` / `max` / `argmin` / `argmax`；`percentile` 与分位数；`axis` 参数到底在"消哪一维"；NaN 处理（`nanmean` / `nanstd`）。

**A.4 pandas Series 与 DataFrame 基础**
Series 的索引对齐；DataFrame 的 `shape` / `columns` / `dtypes` / `info()` / `describe()`；`astype` 类型转换；`rename` 与 `set_index`。

**A.5 pandas 读写与筛选**
`read_csv` 常用参数（`nrows` / `usecols` / `dtype` / `parse_dates`）；`loc` 与 `iloc` 的区别；`query` 表达式筛选；`isin` 与 `between`；缺失值统计 `isna().sum()`。

**A.6 pandas 分组聚合与透视**
`groupby` + `agg` 多函数聚合；`transform` 与 `apply` 的差别（为什么目标编码要用 `transform`）；`pivot_table` 做交叉表；`value_counts` 看类别分布。

**A.7 pandas 合并与连接**
`merge` 的 `how`（inner/left/right/outer）与 `on` / `left_on` / `right_on`；`concat` 的 `axis=0/1`；`join` 与索引对齐；合并后行数变多的排查方法（一对多导致的行膨胀）。

**A.8 matplotlib 常见图表与样式**
折线 / 散点 / 直方图 / 箱线图 / 柱状图的绘制；`figsize` / `dpi` / `bbox_inches="tight"` 的用途；中文字体与负号显示；`plt.close()` 防止内存泄漏。

**A.9 三件套的分工与数据流**
【Mermaid：原始 CSV → pandas 清洗 → NumPy 矩阵 → matplotlib 出图 → 工件落盘】

---

## 附录B：scikit-learn / XGBoost / LightGBM 速查

> 定位：树模型与通用机器学习 API 的对照表。重点不是参数大全，而是"同一个动作在三个库里分别怎么写"。

**B.1 估计器 API 三件套**
`fit` / `predict` / `predict_proba` / `score` 的统一约定；`n_jobs` 与 `random_state` 该不该固定；为什么"换库不换 API"让集成变简单。

**B.2 预处理：缩放、编码、缺失值**
`StandardScaler` / `MinMaxScaler` / `RobustScaler` 的适用场景；`OneHotEncoder` 与 `OrdinalEncoder`；`SimpleImputer` 的 `strategy` 选择；**所有预处理都必须在训练折上 fit**（防泄漏）。

**B.3 交叉验证与网格搜索**
`KFold` / `StratifiedKFold` / `GroupKFold` / `TimeSeriesSplit` 的选择表；`cross_val_score` 与 `cross_val_predict`（后者才产出 OOF）；`GridSearchCV` / `RandomizedSearchCV` 的 `scoring` 与 `refit`。

**B.4 管道与列变换**
`Pipeline` 把预处理和模型串起来；`ColumnTransformer` 对不同列用不同处理；为什么管道能让"预处理只 fit 一次"变成默认行为。

**B.5 评估指标速查**
分类：`roc_auc_score` / `log_loss` / `f1_score` / `accuracy_score`；回归：`mean_squared_error` / `mean_absolute_error` / `mean_squared_log_error`；排序：`ndcg_score` / `average_precision_score`。

**B.6 XGBoost 核心参数与早停**
`n_estimators` / `max_depth` / `learning_rate` / `subsample` / `colsample_bytree` / `reg_lambda`；`early_stopping_rounds` 与 `eval_set` 的正确用法；`tree_method="hist"` 与 GPU。

**B.7 LightGBM 核心参数与类别特征**
`num_leaves` 与 `max_depth` 的换算关系；`min_child_samples`；`categorical_feature` 直接吃类别列的优势；`device="gpu"` 的收益与坑。

**B.8 CatBoost 与 GPU 加速要点**
`cat_features` 自动编码；`task_type="GPU"`；`verbose` 控制；为什么它在类别特征多的表格赛里常是"免费涨点"。

**B.9 三者怎么选**
【Mermaid：数据规模 × 类别特征占比 × 是否需要 GPU → 选 XGB / LGBM / CatBoost 的决策图】

---

## 附录C：AutoGluon 快速参考（Tabular / TimeSeries）

> 定位：本书"自动基线"的主力工具。只保留产出合规工件所需的调用，以及必须人来判断的那几个开关。

**C.1 Tabular 快速开始**
`TabularPredictor(label=..., eval_metric=..., path=...)`；`fit(train_data, time_limit=...)`；`predict` / `predict_proba`；`leaderboard()` 看内部集成结果。

**C.2 presets 档位与 time_limit 的取舍**
`best_quality` / `high_quality` / `medium_quality` / `fast` 四档的差异；`time_limit` 设太小会导致"基线还没训完就结束"；`num_bag_folds` / `num_stack_levels` 对 OOF 质量的影响。

**C.3 产出 OOF 与测试预测工件的标准写法**
用**人定的固定折**算 OOF（而不是 AutoGluon 内部随便切的折）；把 `predict_proba` 结果按 `train_oof_[MODEL]_[VERSION].npy` 规范落盘；同时保存测试预测与配置快照。

**C.4 内部集成机制与"它已经 stacking 过了"**
AutoGluon 自带多层 stacking 与 bagging；人工 stack 的价值在于引入它之外的模型家族；不要重复造它已经做好的那层。

**C.5 TimeSeries 快速开始**
`TimeSeriesPredictor(prediction_length=..., target=..., eval_metric=..., path=...)`；长格式数据要求（`item_id` / `timestamp` / `target`）；`fit(..., time_limit=...)` 与 `predict`。

**C.6 时序基线工件与零样本对照**
AutoGluon-TimeSeries 基线 + Chronos / TimesFM 零样本预测并列存档；两者误差模式不同，是后续融合的多样性来源。

**C.7 常见坑**
内存不足（降 presets / 减 bag folds）；时间预算被"第一个模型"吃光；内部折与外部 CV 折不一致导致 OOF 不可比；GPU 不可用时自动回退 CPU 的静默行为。

---

## 附录D：PyTorch / HuggingFace Transformers 速查

> 定位：图像与文本章节的代码底座。重点是"冻结骨干 + 轻量头"这条主路径，以及训练循环的最小骨架。

**D.1 Tensor 创建、索引与设备迁移**
`torch.tensor` / `torch.zeros` / `torch.randn`；`view` vs `reshape`；`.to(device)` 与 `.cuda()`；`detach()` / `item()` 的区别；默认 dtype 与 float32 的显存账。

**D.2 Dataset 与 DataLoader**
`Dataset` 的 `__len__` / `__getitem__`；自定义 collate；`DataLoader` 的 `batch_size` / `shuffle` / `num_workers` / `pin_memory`；类别不平衡时的采样策略。

**D.3 nn.Module 与训练循环骨架**
`__init__` 定义层、`forward` 定义数据流；`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` 四步；`model.train()` / `model.eval()` 的差别（Dropout / BN）。

**D.4 优化器、学习率调度与混合精度**
`AdamW` 与 `SGD` 的取舍；`torch.optim.lr_scheduler` 的 warmup + 余弦 / 线性衰减；`torch.cuda.amp` 自动混合精度；梯度裁剪 `clip_grad_norm_`。

**D.5 Transformers 模型加载与分词器**
`AutoModel` / `AutoModelForSequenceClassification` / `AutoTokenizer`；`max_length` 与截断策略；`padding` / `truncation` / `return_tensors="pt"`；`attention_mask` 为什么必须传。

**D.6 冻结骨干 + 轻量头的标准写法**
冻结前 N 层（`requires_grad_(False)`）；只训 head；何时解冻末尾几层；学习率分层（骨干小、头大）；小数据上"先训头再解冻"的两段式。

**D.7 保存、加载与推理一致性**
`state_dict` 与整模型保存的取舍；推理必须 `model.eval()` + `torch.no_grad()`；TTA 后概率要与验证集同口径校准；图像预处理（尺寸 / 归一化 / 通道顺序）必须与训练一致。

---

## 附录E：表格基础模型速查（TabPFN 等）

> 定位：零调参基线的来源。回答"什么时候该用它、什么时候不该用"。

**E.1 TabPFN 是什么、什么时候用**
Prior-Data Fitted Network 的直觉（"作业题都做过一遍了"）；中小规模（几千到几万行）表格上的零调参强基线；与树模型误差模式不同，适合做多样性来源。

**E.2 TabPFN 最小可用示例**
【Python 代码示例：TabPFNClassifier / TabPFNRegressor 的 fit + predict_proba，并产出合规工件】

**E.3 与树模型的互补性**
OOF 相关性检查（< 0.95 才有集成价值）；"一个不调参的模型跑赢 36 模型手工堆叠"的真实案例与它的边界。

**E.4 其他表格基础模型与适用规模**
不同基础模型对行数 / 类别特征 / 缺失值的容忍度对照表；超出规模时的降级策略（采样 + 树模型兜底）。

**E.5 使用前提：规则是否允许预训练模型**
部分比赛限制预训练模型与外部数据；先读 `Rules` 再决定用不用；把模型名与版本写进配置而不是硬编码。

---

## 附录F：Kaggle MCP 配置与安全实践

> 定位：让 Agent 能持续操作平台，同时把"不可逆"关在笼子里。这一章是全书安全边界的落地页。

**F.1 MCP 是什么、解决什么问题**
Model Context Protocol 的直觉（Agent 的"手和脚"）；没有 MCP 只能聊天、有了 MCP 能下载 / 训练 / 落盘 / 提交；为什么"全包"是双刃剑。

**F.2 安装与配置**
配置文件位置与最小可用配置片段；`mcp` 服务器启动命令；连通性自检（先跑 `list_experiments` 这类只读工具）。

**F.3 工具清单与权限白名单**
`read_data` / `write_artifact` / `train_model` / `predict` / `submit_dryrun` / `submit` / `list_experiments` / `compute_oof_corr` / `generate_report`；`forbidden_tools` 里为什么必须有 `shell_exec`。

**F.4 预览-确认双调用与提交预算**
所有写操作默认 `dry_run=true`；人确认后才 `false`；`daily_submit_limit` 的设定依据（正式赛约 5 次 / Playground 约 9 次，GMT 午夜重置）。

**F.5 审计日志与不可逆操作**
`mcp_audit.jsonl` 记录每次调用；`allow_destructive: false` 默认关闭删除；事后追溯"Agent 到底动了什么"。

**F.6 常见故障排查**
下载失败（门控权重 / 网络）；提交被拒（格式 / 额度）；工件写不进去（路径 / 权限）；超时（time_limit 与平台时限）。

---

## 附录G：Kaggle 评估指标速查（含“Agent 容易在哪个指标上骗你”的提示）

> 定位：指标是"人不可让渡的判断"的第一项。这张表既给公式，也给"这个指标在什么情况下会骗你"。

**G.1 回归指标**
RMSE / MAE / RMSPE / MAPE / R² 的公式与适用场景；异常值对 RMSE 的放大效应；真实值接近 0 时 MAPE 爆炸的原因与兜底。

**G.2 分类指标**
AUC / LogLoss / Accuracy / F1 / 精确率-召回率的取舍；类别不平衡下 Accuracy 为什么失效；AUC 对概率校准不敏感、LogLoss 敏感。

**G.3 排序与多标签指标**
MAP / NDCG / 列平均 AUC（macro AUC）/ F2；多标签任务里"列平均"与"整体平均"的差别；稀有类别被平均稀释的问题。

**G.4 加权与自定义指标怎么读**
加权 MAE / 加权 R² 的权重从哪来；官方指标里"补偿不平衡"的设计意图；读不懂权重就先别调模型。

**G.5 Agent 容易在哪个指标上骗你**
【Mermaid：指标 → 常见自欺方式 → 对应检查动作 的对照图】；用 Accuracy 掩盖不平衡、用随机切分抬高 CV、用 TTA 后概率套验证集阈值、用公开榜当验证集。

**G.6 指标与验证方案的搭配表**
指标类型 × 数据分布 × 时间性 → 推荐的验证切分；一张表查完"该用什么指标、该怎么切"。

---

## 附录H：工件管理与实验日志规范（完整版）

> 定位：第 2 章工件规范的完整落地版。这一章是"可组合、可追溯、可审计"三件事的全部细节。

**H.1 目录结构（完整版）**
`artifacts/`（oof / test_preds / logs / configs / submissions / ensembles）+ `data/`（只读）+ `src/` + `configs/` + `notebooks/` + `scripts/` 的完整树；为什么 `artifacts/` 要进 `.gitignore` 或用 DVC。

**H.2 命名约定与版本号规则**
`train_oof_{模型}_{版本}.npy` / `test_preds_{模型}_{版本}.npy` / `exp_{模型}_{版本}.jsonl` / `cfg_{模型}_{版本}.yaml` / `submission_{模型}_{版本}.csv`；版本号三位数字严格递增；多任务后缀 `_cls` / `_reg`。

**H.3 元数据 Schema（完整字段）**
【Python 代码示例：Pydantic `ArtifactMetadata` 全字段（身份 / 验证协议 / 性能 / 工件路径 / 数据血统 / 变更描述 / 时间与来源）】

**H.4 JSON Lines 日志写入与读取**
一行一个实验的追加写；读取时按模型 / 版本 / 时间过滤；日志损坏（半行写入）的容错处理。

**H.5 折索引哈希与验证协议校验**
`cv_fold_indices_hash` 的生成与比对；扫描全部日志揪出"验证协议不一致"的实验；为什么哈希是防作弊的锚点。

**H.6 工件清理、归档与版本管理**
过期工件的归档策略；DVC / MLflow 管理大文件；清理前必须确认"这个工件没有被任何集成引用"。

**H.7 一次实验的完整检查清单**
从读协议 → 固定折 → 训练 → 落盘 → 写日志 → 自检 → 人审的逐项勾选表；任何一项没打勾就不算"合规工件"。


## 更新历史

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-11-25 | v2.2 | 附录细纲由单行条目扩充为 x.y 二级小节：附录A（A.1–A.9）、B（B.1–B.9）、C（C.1–C.7）、D（D.1–D.7）、E（E.1–E.5）、F（F.1–F.6）、G（G.1–G.6）、H（H.1–H.7），每个附录补定位说明与小节要点 |
| 2026-11-25 | v2.1 | 添加“**附录D**：PyTorch / HuggingFace Transformers 速查”，其它附录编号顺延 | 
| 2026-11-22 | v2.0 | 比赛解析章节全部替换为 2024—2026 年比赛：表格（S6E3 客户流失 / S5E10 道路事故 / S6E1 学生成绩 / NeurIPS Open Polymer 2025）、时序（S5E1 贴纸销售 / S5E10 道路事故 / Jane Street 2024—2025）、图像（CSIRO Image2Biomass 2025—2026 / RSNA Knee 2026 / Image Matching 2025）、文本（Jigsaw Agile Community Rules 2025 / LLM Prompt Recovery 2024 / Make Data Count 2025）。每章保留 VibeKaggle 的“自动基线工件 → 人设定验证纪律 → Agent 批量实验 → 人审查工件并选择”循环，并引用具体获胜方案中的工件管理和人机分工细节 |