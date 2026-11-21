# 写作要求


+   面向编程语言和深度学习入门者，不要假设有其他知识
+   语言风格详细直白，关键概念用大白话解释。可以添加一些废话，使得文本更加流畅和自然
+	需要配套带有详细注释的 Python 代码，关键概念配有 mermaid 图表
+	编写每章之前先调用百度搜索，收集至少 20 条结果
+	如果某节需要调用大模型，确保正文中包含调用记录，一问一答形式，确保用户输入和模型输出包含在三个反引号（```）内，如果模型有思考过程，一并带上，示例如下：

	我们首先让模型翻译一段话：
	
	```
	请把以下句子翻译成英文：死亡如风，常伴吾身。
	```
	
	模型迅速给出了翻译：
	
	```
	> （此处填写思考过程）

	Death is like the wind; always by my side.
	```
	
	接着我们让模型写一段 Python 代码：
	
	```
	再写一段 Python 函数计算锤子线指标，函数接受参数`df`包含 OHLCV 数据，接受参数`metric_col`指定结果列名，接受参数`in_place`指定是否原地修改，并返回带有结果列的数据帧。
	```
	
	以下是模型输出的代码：
	
	```python
	import pandas as pd
	import numpy as np


	def hammer(
		df: pd.DataFrame,
		metric_col: str = "hammer",
		in_place: bool = False
	) -> pd.DataFrame:
		"""
		计算锤子线指标。

		参数
		----------
		df : pd.DataFrame
			包含 OHLCV 数据的数据帧，至少需要 open/high/low/close 列。
			列名大小写不敏感。
		metric_col : str, default "hammer"
			结果列名。
		in_place : bool, default False
			是否原地修改 df。为 True 时修改并返回原数据帧；
			为 False 时复制后修改并返回新数据帧。

		返回
		-------
		pd.DataFrame
			带有锤子线结果列的数据帧，结果列类型为 bool。
		"""
		required = ["open", "high", "low", "close"]

		# 支持大小写列名，例如 Open/High/Low/Close
		col_map = {}
		for req in required:
			matches = [c for c in df.columns if str(c).lower() == req]
			if not matches:
				raise ValueError(f"数据帧缺少必要列: {req}")
			col_map[req] = matches[0]

		out = df if in_place else df.copy()

		o = out[col_map["open"]]
		h = out[col_map["high"]]
		l = out[col_map["low"]]
		c = out[col_map["close"]]

		body = (c - o).abs()
		total_range = h - l
		lower_shadow = np.minimum(o, c) - l
		upper_shadow = h - np.maximum(o, c)

		out[metric_col] = (
			(total_range > 0)
			& (body > 0)
			& (body <= total_range * 0.3)
			& (lower_shadow >= 2 * body)
			& (upper_shadow <= body)
		)

		return out
	```
	
	最后让模型简单赏析以下这段话吧：
	
	```
	请简要赏析这段话：海阔老舍跃，天高科比飞。商鞅知马力，比干见人心。
	```
	
	这里是模型给出的赏析：
	
	```
	> （此处填写思考过程）
	
	这段话是典型的网络“人名梗”拼贴，戏仿传统格言对联，妙在“一本正经地胡说八道”。

	1. **化用来源**  
	   - “海阔老舍跃，天高科比飞”化自“海阔凭鱼跃，天高任鸟飞”，把“鱼”“鸟”换成作家“老舍”和球星“科比”，造成语义错位。  
	   - “商鞅知马力，比干见人心”化自“路遥知马力，日久见人心”，用历史人物“商鞅”“比干”替换“路遥”“日久”，并借商鞅变法、比干剖心等典故制造双关。

	2. **手法与趣味**  
	   它把名人专名硬嵌入熟语框架，形成陌生化和荒诞感；前联偏无厘头，后联稍带典故双关。读者先认出原句，再被替换词“闪一下”，幽默就产生了。

	3. **效果与局限**  
	   作为网络文字游戏，它机智、戏谑、有解构意味；但平仄、对仗并不工整，意义也主要是“错位搞笑”，不宜当作严格对联或格言看。总体是“熟语框架＋名人拼贴＋典故双关”的趣味段子。
	```

	
	
