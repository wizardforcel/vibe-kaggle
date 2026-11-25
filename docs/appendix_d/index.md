# 附录D：PyTorch / HuggingFace Transformers 速查

> 图像、文本任务的主力深度学习库。PyTorch 当框架，Transformers 当模型库。

---

## D.1 PyTorch 速查

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- 张量 ----
x = torch.tensor([1.0, 2.0, 3.0])   # 张量
x = x.cuda() if torch.cuda.is_available() else x

# ---- 模型（继承 nn.Module） ----
class DinoHead(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.head(x).squeeze(-1)

model = DinoHead(in_dim=768)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---- 损失与优化器 ----
criterion = nn.MSELoss()            # 回归
# criterion = nn.CrossEntropyLoss() # 分类
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- 训练步骤 ----
model.train()
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    opt.step()

# ---- 推断 ----
model.eval()
with torch.no_grad():
    pred = model(xb.to(device)).cpu().numpy()
```

**冻结骨干**：
```python
for p in model.backbone.parameters():  # 锁住 backbone，只训 head
    p.requires_grad = False
```

---

## D.2 HuggingFace Transformers 速查

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "microsoft/deberta-v3-base"

# ---- 加载 ----
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# ---- 分词 ----
enc = tokenizer(["hello world", "goodbye world"], truncation=True,
                padding="max_length", max_length=256, return_tensors="pt")

# ---- 前向 ----
out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
logits = out.logits   # (batch, num_labels)
probs = torch.softmax(logits, dim=-1)
```

**微调基线（文本分类）**：
```python
model.fit(...)  # 见全文第 16 章文本基线
```

**保存/加载**：
```python
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")
# 加载
model = AutoModelForSequenceClassification.from_pretrained("my_model")
```

下一附录是表格基础模型速查。