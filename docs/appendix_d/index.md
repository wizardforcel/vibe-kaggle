# 附录D：PyTorch / HuggingFace Transformers 速查

> 本附录是图像与文本章节的代码底座。重点不在"训一个 SOTA"，而在**冻结骨干 + 轻量头**这条图像/文本章的主路径，以及训练循环的最小骨架。入门者只需看懂"怎么把一个预训练模型变成自己的模型"即可。

---

## D.1 Tensor 创建、索引与设备迁移

PyTorch 的核心对象是 Tensor，翻译成人话就是"带 GPU 支持的 NumPy 数组"。

```python
import torch

# ---- 创建 ----
t = torch.tensor([1.0, 2.0, 3.0])              # 从列表/数组创建
z = torch.zeros(2, 3)                           # 全 0 矩阵 (2,3)
r = torch.randn(4, 5)                           # 标准正态分布随机矩阵
a = torch.arange(10)                            # 等差序列

# ---- 索引 ----
t[0]                                            # 第 0 个元素
t[1:3]                                          # 切片
t[t > 1]                                        # 布尔掩码：大于 1 的元素

# ---- 形状 ----
t.view(3, 1)                                    # 改形状（元素总数必须一致）
t = t.reshape(-1)                               # 拉平成一维

# ---- 设备迁移 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = t.to(device)                                # 把张量搬到 GPU/CPU
```

设备迁移是新手最常漏的一步：模型在 GPU 上，数据却在 CPU 上，会直接报错。跑任何训练/推理前，先 `device = torch.device(...)`，模型和数据都 `.to(device)`。

还有一个常见混淆：`view` 和 `reshape` 都改形状，但 `view` 要求内存连续，不连续时报错；`reshape` 会自动处理，更稳妥，优先用 `reshape`。

**什么时候会用到**：把图像/文本转成数字矩阵喂给模型、把标签和特征搬到同一设备、检查中间张量的形状。

---

## D.2 Dataset 与 DataLoader

PyTorch 读数据分两步：先定义一个 `Dataset`（怎么读一条），再用 `DataLoader`（怎么分批、怎么打乱）。

```python
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, df):
        self.x = torch.tensor(df.values, dtype=torch.float32)
        self.y = torch.tensor(df["target"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]         # 返回一条（特征，标签）

dataset = CSVDataset(train_df)
loader = DataLoader(
    dataset,
    batch_size=64,          # 一批多少条
    shuffle=True,           # 训练集打乱，验证集/测试集设 False
    num_workers=2,          # 多进程加载，0 表示主进程
    pin_memory=True,        # 加速 GPU 传输（GPU 环境下建议）
)
```

`__getitem__` 返回一条数据，`DataLoader` 负责把一批条打成 batch、按需打乱、多进程预加载。训练时 `for x, y in loader:` 就能拿到一批数据。

类别不平衡时，`shuffle=True` 不够，还需要加权采样器（`WeightedRandomSampler`），让少数类被抽到的概率更高。

**什么时候会用到**：图像/文本章读自己的数据集、做验证集划分、处理类别不平衡。

---

## D.3 nn.Module 与训练循环骨架

```python
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)   # 定义层

    def forward(self, x):
        return self.fc(x)                      # 定义数据流

model = SimpleNet(10, 1).to(device)
criterion = nn.MSELoss()                       # 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):                         # 训练循环
    model.train()                              # 进入训练模式（启用 Dropout/BN）
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()                  # 梯度清零
        pred = model(x)                        # 前向
        loss = criterion(pred, y)              # 算损失
        loss.backward()                        # 反向传播，算梯度
        optimizer.step()                       # 更新权重
```

训练循环只有四步，顺序不能错：**清零梯度 → 前向 → 算损失 → 反向 → 更新**。漏掉 `zero_grad()`，梯度会累积到上一批，结果越来越怪。

`model.train()` 和 `model.eval()` 是一对开关：训练时启用 Dropout / BatchNorm 的训练行为，推理/验证时关掉。推理前一定要加 `model.eval()`，否则 Dropout 会让同一张图预测两次结果不一样。

**什么时候会用到**：写任何 PyTorch 模型，都是这个骨架 + 自己的层。

---

## D.4 优化器、学习率调度与混合精度

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 混合精度（节省显存，加速训练）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for x, y in loader:
    optimizer.zero_grad()
    with autocast():                     # 自动选 float16/float32
        pred = model(x)
        loss = criterion(pred, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

- `AdamW` 是目前默认首选，带权重衰减（weight_decay）防过拟合；
- `CosannealingLR` 把学习率按余弦曲线慢慢降到 0，常配 warmup（前几轮先小后大）；
- 混合精度 `autocast` + `GradScaler` 让显存占用大概减半、速度加快，GPU 环境下强烈建议开。

学习率调度怎么选：数据少就小学习率 + 长 warmup；数据多就大学习率 + 短 warmup。不确定时先 `1e-3`，看 loss 曲线再调。

**什么时候会用到**：训练任何神经网络，决定"怎么更新权重、怎么控制学习率"。

---

## D.5 Transformers 模型加载与分词器

```python
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"   # 或者 "Qwen/Qwen3-4B" 等
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# 分词
texts = ["你好世界", "另一段文本"]
inputs = tokenizer(
    texts,
    padding=True,            # 补齐到同长度
    truncation=True,         # 超长截断
    max_length=128,          # 最大长度
    return_tensors="pt",     # 返回 PyTorch tensor
).to(device)

outputs = model(**inputs)
logits = outputs.logits      # 模型原始输出
```

分词器的三个关键参数：
- `padding=True`：把不同长度的句子补齐成一样长，才能组成 batch；
- `truncation=True`：超过 `max_length` 的截掉，防止炸显存；
- `return_tensors="pt"`：直接给 PyTorch tensor，不用手动转。

`num_labels=2` 是二分类；回归任务用 `AutoModelForSequenceClassification` 并把 `num_labels=1`，用 `MSELoss`。

**什么时候会用到**：文本章的任何微调，都离不开 `from_pretrained` + `tokenizer` 这两行。

---

## D.6 冻结骨干 + 轻量头的标准写法

这是全书图像/文本章的**主路径**：不用从零训一个大模型，只在预训练骨干后面接一个小"头"（线性层），只训头。

```python
from transformers import AutoModel
import torch.nn as nn

class DinoHead(nn.Module):
    def __init__(self, backbone_name, out_dim=1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        # 冻结骨干：不更新参数
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        # 轻量头：只训这个
        self.head = nn.Linear(self.backbone.config.hidden_size, out_dim)

    def forward(self, x):
        with torch.no_grad():          # 骨干不算梯度，省显存
            feat = self.backbone(x).last_hidden_state[:, 0]  # CLS token
        return self.head(feat)

# 两段式：先训头，再解冻末尾几层
model = DinoHead("google/vit-base-patch16-224").to(device)
optimizer = optim.AdamW(model.head.parameters(), lr=1e-3)  # 只优化头

# ... 训几个 epoch 后 ...
# 解冻骨干末尾几层，小学习率继续训
for param in model.backbone.encoder.layer[-2:].parameters():
    param.requires_grad_(True)
optimizer = optim.AdamW([
    {"params": model.backbone.encoder.layer[-2:].parameters(), "lr": 1e-5},
    {"params": model.head.parameters(), "lr": 1e-3},
], lr=1e-3)
```

**学习率分层**：骨干用小学习率（1e-5），头用大学习率（1e-3），避免把预训练好的通用特征冲掉。

冻结的判断标准（第 12 章已展开）：数据只有几百到几千张 → 冻结全部；验证指标不再动且训练损失压平 → 解冻末尾 2~4 层；数据十万张以上且域差异大 → 全量微调。

**什么时候会用到**：第 12–15 章图像章、第 16–19 章文本章的每一次微调。

---

## D.7 保存、加载与推理一致性

```python
# ---- 保存（推荐 state_dict，不要存整个模型） ----
torch.save(model.state_dict(), "checkpoint.pt")

# ---- 加载 ----
model = DinoHead(...)
model.load_state_dict(torch.load("checkpoint.pt", map_location=device))
model.eval()

# ---- 推理（必须 eval + no_grad） ----
with torch.no_grad():
    model.eval()
    pred = model(x.to(device))
```

两个铁律：
1. **存 `state_dict`，不存整个模型对象**。存整个模型会绑定代码结构，换个环境加载就报错；`state_dict` 只存参数，跨平台更稳。
2. **推理时 `model.eval()` + `torch.no_grad()`**。不加 `eval()`，Dropout 还在工作，同一张图两次预测不同；不加 `no_grad()`，显存白白浪费在算梯度上。

推理一致性还有一点：训练时怎么预处理（图像尺寸 / 归一化 / 通道顺序 / 文本分词长度），推理时必须**一模一样**。TTA（测试时增强）后的概率，要和验证集用同一套概率口径，否则阈值就错位了（第 12 章有专门讨论）。

**什么时候会上**：训完模型存档、加载别人开源的权重、写推理服务（第 23 章 FastAPI 接口）。

推理前最容易漏的一步就是模型状态，可以直接问：

> **提示词**：推理时忘了 model.eval()，会出什么问题？能简单解释一下吗？

模型会这样答：

> **AI 输出**
>
> （思考过程）用户要简单解释，不要堆术语。核心是 Dropout 和
> BatchNorm 这两层。给一个"训练 vs 推理"的对比表，再给一句话
> 结论。
>
> 简答：会让 Dropout 在推理时也"随机丢弃神经元"，同一张图两次
> 预测结果不一样，而且概率分布会被拉偏，指标虚高。
>
> 记住一句话：**`model.eval()` 是推理时的"必须动作"，不加就等于
> 还在训练。**

---

## 小结

PyTorch / Transformers 在这本书里只做一件事：**冻结骨干 + 轻量头**：

- `Tensor` / `Device`：数据怎么放、往哪搬；
- `Dataset` / `DataLoader`：怎么分批、怎么打乱；
- `nn.Module` + 四步训练循环：怎么定义模型、怎么更新；
- `AdamW` / 调度器 / 混合精度：训得快不快、稳不稳；
- `AutoModel` + `AutoTokenizer`：怎么加载预训练模型；
- 冻结骨干 / 学习率分层 / 两段式：怎么不动通用特征、只适配自己的任务；
- `state_dict` / `eval()` / `no_grad()`：怎么存、怎么推理不出错。

不需要一次记住，用到时回来查。下一个附录是表格基础模型 TabPFN 等——零调参基线的另一来源。