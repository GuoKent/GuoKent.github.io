---
title: Attention 算法
date: 2024-10-03 00:00:00
math: true
tags:
- 深度学习
- 算法
- 基础
categories:
- 算法杂记
alias:
- deeplearning/attention/
---

## Self-Attention
$$
\begin{aligned}
& Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V \\
& \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T} { \sqrt{d_k} }\right)V \\
& X.\text{shape}: [B, T, D], \quad W.\text{shape}: [D, D], \quad d_k = D
\end{aligned}
$$

### 理论解释为什么 Attention 有效
假设我们有 5 个单词 `{cat, milk, it, sweet, hungry}`, 每个单词用一个向量表示:
$$
\begin{aligned}
\text{sweet} &= (..., 0, 4, ...)\\
\text{milk} &= (..., 1, 3, ...)\\
\text{it} &= (..., 2, 2, ...)\\
\text{cat} &= (..., 2, 2, ...)\\
\text{hungry} &= (..., 4, 0, ...)
\end{aligned}
$$

假设词向量的维度 dim=4，第一列数字(第 1 个头)代表这个单词关于`状态`的属性，值越高代表该单词与`hungry`越相关; 第二列数字(第 2 个头)代表这个单词关于`味道`的属性，值越高代表该单词与`sweet`越相关。

#### 例子 1
现在让我们考虑 Attention 算法中，计算`状态`这部分的头。假设我们正在处理一个句子 **"The cat drank the milk because it was sweet."**，其中包含了 cat、milk、it、sweet 这四个单词(暂且忽略其余不相关单词，单词按词序组成矩阵)。此时在 Self-Attention 算法中的 Q、K、V 矩阵为:
$$
Q=K=V=
\begin{bmatrix}
  ... & 2 & 2 & ...\\ 
  ... & 1 & 3 & ...\\ 
  ... & 2 & 2 & ...\\ 
  ... & 0 & 4 & ...
\end{bmatrix}
\begin{matrix}
  \text{cat} \\ 
  \text{milk} \\ 
  \text{it} \\ 
  \text{sweet}
\end{matrix}
$$

现在我们计算 Attention 分数(为了方便理解，`...` 部分用 0 代替):

$$
Q \cdot K^T=
\begin{array}{cccccc}
  & \text{cat} & \text{milk} & \text{it} & \text{sweet} \\
\text{cat} & 8 & 8 & 8 & 8 \\ 
\text{milk} & 8 & 10 & 8 & 12 \\ 
\text{it} & 8 & 8 & 8 & 8 \\ 
\text{sweet} & 8 & 12 & 8 & 16 \\
\end{array}
\\
Softmax(\frac{Q \cdot K^T}{\sqrt{d}}) \cdot V = 
\begin{bmatrix}
  ... & 0.625 & 1.375 & ...\\ 
  ... & 0.089 & 1.911 & ...\\ 
  ... & 0.625 & 1.375 & ...\\ 
  ... & 0.010 & 1.990 & ...
\end{bmatrix}
\begin{matrix}
  \text{cat} \\ 
  \text{milk} \\ 
  \text{it} \\ 
  \text{sweet}
\end{matrix}
$$

之后，我们得到这 4 个单词的新 embedding 为:
$$
\begin{aligned}
\text{sweet} &= (..., 0, 4, ...) \rightarrow (..., 0.010, 1.990, ...)\\
\text{milk} &= (..., 1, 3, ...) \rightarrow (..., 0.089, 1.911, ...)\\
\text{it} &= (..., 2, 2, ...) \rightarrow (..., 0.625, 1.375, ...)\\
\text{cat} &= (..., 2, 2, ...) \rightarrow (..., 0.625, 1.375, ...)\\
\text{hungry} &= (..., 4, 0, ...) 
\end{aligned}
$$

通常情况下， 在英文中 it 既可以指代 cat 又可以指代 milk，因此 it 和 cat 的相似度与 it 和 milk 的相似度相同，即：
$$
\text{sim(it, milk)} = 1 \times 2 + 3 \times 2 = 8 \\
\text{sim(it, cat)}  = 2 \times 2 + 2 \times 2 = 8
$$
之后，模型学习了句子 **"The cat drank the milk because it was sweet."**，这个句子中 it 指代 milk，通过 Attention 算法后得到了新的词 embedding，这时 it 在词向量表达上更加靠近 milk：
$$
\text{sim(it, milk)} = 0.0625 \times 0.089 + 1.375 \times 1.911 = 2.68 \\
\text{sim(it, cat)}  = 0.0625 \times 0.0625 + 1.375 \times 1.375 = 2.28
$$

#### 例子 2
再来看另一种情况，对于另一个句子 **"The cat drank the milk because it was hungry."**，这个句子中 it 指代 cat，我们同样运用 Attention 算法，得到新的词 embedding：
$$
Q=K=V=
\begin{bmatrix}
  ... & 2 & 2 & ...\\ 
  ... & 1 & 3 & ...\\ 
  ... & 2 & 2 & ...\\ 
  ... & 4 & 0 & ...
\end{bmatrix}
\begin{matrix}
  \text{cat} \\ 
  \text{milk} \\ 
  \text{it} \\ 
  \text{hungry}
\end{matrix}
$$
计算 Attention 分数：
$$
Q \cdot K^T=
\begin{array}{cccccc}
  & \text{cat} & \text{milk} & \text{it} & \text{hungry} \\
\text{cat} & 8 & 8 & 8 & 8 \\ 
\text{milk} & 8 & 10 & 8 & 4 \\ 
\text{it} & 8 & 8 & 8 & 8 \\ 
\text{hungry} & 8 & 4 & 8 & 16 \\
\end{array}
\\
Softmax(\frac{Q \cdot K^T}{\sqrt{d}}) \cdot V = 
\begin{bmatrix}
  ... & 1.125 & 0.875 & ...\\ 
  ... & 0.609 & 1.391 & ...\\ 
  ... & 1.125 & 0.875 & ...\\ 
  ... & 1.999 & 0.001 & ...
\end{bmatrix}
\begin{matrix}
  \text{cat} \\ 
  \text{milk} \\ 
  \text{it} \\ 
  \text{hungry}
\end{matrix}
$$

之后我们得到 4 个单词的新 embedding：
$$
\begin{aligned}
\text{hungry} &= (..., 4, 0, ...) \rightarrow (..., 1.999, 0.001, ...)\\
\text{milk} &= (..., 1, 3, ...) \rightarrow (..., 0.609, 1.391, ...)\\
\text{it} &= (..., 2, 2, ...) \rightarrow (..., 1.125, 0.875, ...)\\
\text{cat} &= (..., 2, 2, ...) \rightarrow (..., 1.125, 0.875, ...)\\
\text{sweet} &= (..., 0, 4, ...) \\
\end{aligned}
$$
此时 it 的词向量更加接近 cat:
$$
\text{sim(it, milk)} = 1.125 \times 0.609 + 0.875 \times 1.391 = 1.90 \\
\text{sim(it, cat)}  = 1.125 \times 1.125 + 0.875 \times 0.875 = 2.03
$$


### 为什么要 scaling？
scaling 目的是解决数值稳定性问题，从而提高训练的效率和性能。当 Q，K 的维度 $d_k$ 很大时， $q_i,k_i$ 的点积值可能变得很大。点积值越大，输入到 softmax 函数中的数值范围越广，可能会导致以下问题：
- **softmax 的梯度变得极小**: softmax 函数对大数值非常敏感，极大值会导致其他位置的权重几乎为 0，从而产生数值不稳定性。
- **模型训练变得困难**: 梯度消失问题会使得模型难以学习。

### 为什么是 $d_k$ 而不是其他数？
对于输入特征 $X$，其元素通常服从均值为 0、方差为 1 的标准正态分布。经过点积计算 $QK^T$ 后，由于 $q_i,k_i$ 的点积是 $d_k$ 个独立随机变量的和，所以方差会变为 $d_k$，除以 $\sqrt{d_k}$ 可以让方差重新变为 1。
- 如果直接除以 $d_k$，方差变为 $1/d_k$，分布过于集中，让 softmax 的值趋于均匀分布，会弱化注意力机制的效果
- 如果除以 $\sqrt[3]{d_k}$，会让方差仍然较大，可能会导致数值不稳定和训练困难

### 其余 scale 处理
1. 归一化 Attention 分数，放缩到[0,1]范围
    $$
    score=\frac{QK^T}{||QK^T||}
    $$

2. 温度参数：$d_k$ 换成常数

3. 预归一化
    $$
    Q'=\frac{Q}{||Q||}, K'=\frac{K}{||K||}, V'=V
    $$

4. 长序列放缩，长序列导致 Attention 分数范围增大，Softmax 失效
    $$
    softmax(\frac{QK^T}{\sqrt{d_k} \cdot \sqrt{T}})V\\
    $$

### 代码手撕
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, input_dim, dim_k=512):
        super().__init__()
        self.norm_factor = 1 / math.sqrt(dim_k)
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_k)

    def forward(self, x, mask=None):
        """
        x.shape: [B, T, D]
        """
        Q, K, V = self.q(x), self.k(x), self.v(x)
        # torch.bmm 输入为 3 维矩阵, 批量相乘, 速度快
        # torch.matmul 输入可为多种矩阵, 更灵活
        score = torch.bmm(Q, K.transpose(1, 2)) * self.norm_factor
        if mask is not None:
            score += mask * -1e9
        return torch.bmm(torch.softmax(score, dim=-1), V)
```

## 多头自注意力
### 为什么要多头？
模型只能学习到一个层面的注意力模式，不能捕捉到输入序列中复杂的多样性关系。仅通过单个头来表示查询、键和值的投影，会限制模型的表达能力。
多头注意力的优势包括：
- **捕捉不同的上下文信息**：每个注意力头可以专注于不同的上下文信息或关系。例如，一个头可以专注于捕捉远距离词语之间的关系，而另一个头可以专注于局部词语之间的关系。
- **提高模型的表达能力**：通过并行计算多个注意力分布，模型能够从多个角度理解同一输入，从而获得**更丰富的语义信息**。
- **提升模型的灵活性和鲁棒性**：多头注意力使得模型能够在多个子空间中进行学习，从而减少单一注意力头可能带来的信息损失。

举例：“The quick brown fox jumped over the lazy dog”。我们希望 Transformer 模型能够理解以下关系：
- **主谓关系**：“fox” 和 “jumped”
- **定语关系**：“quick” 修饰 “fox”
- **位置关系**：“over” 与 “jumped”

对模型来说，不同的头用来学习单词之间的不同关系，比如 "jump" 的头 1 用来学习与"fox"的主谓关系，头 2 用来学习与"over"的位置关系。

### 代码手撕
```python
import torch
import torch.nn as nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, dim_k, num_head=8):
        super().__init__()
        assert dim_k % num_head == 0
        self.dk = dim_k // num_head
        self.head = num_head
        # dk 缩减后放缩因子也要改变
        self.norm_factor = 1 / math.sqrt(self.dk)
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_k)

    def forward(self, x, mask=None):
        """
        x.shape: [B, T, D]
        """
        batch, seqlen, _ = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        # (B, T, D)
        Q, K, V = (
            Q.reshape(batch, seqlen, -1, self.dk).transpose(1, 2),
            K.reshape(batch, seqlen, -1, self.dk).transpose(1, 2),
            V.reshape(batch, seqlen, -1, self.dk).transpose(1, 2),
        )
        # (B, H, T, dk)

        score = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_factor
        if mask is not None:
            score += mask * -1e9
        output = torch.matmul(torch.softmax(score, dim=-1), V)
        output = output.reshape(batch, seqlen, -1)
        return output
```