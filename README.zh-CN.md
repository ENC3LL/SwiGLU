<div align="center">

# ⚡ PyTorch 版 SwiGLU

**开箱即用的单文件 SwiGLU 前馈层：内置融合 Triton 内核，并可自动回退到纯 PyTorch。**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-optional-76B900?logo=nvidia&logoColor=white)](https://github.com/triton-lang/triton)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Single file](https://img.shields.io/badge/single--file-drop--in-blue)](swiglu.py)

[English](README.md) · **简体中文** · [Русский](README.ru.md)

</div>

---

## 这是什么？

一个可直接使用的 **SwiGLU**（Swish 门控线性单元）前馈模块，**LLaMA**、PaLM、Mistral 等主流大语言模型都在使用它。把一个文件复制进项目，`import` 即可使用。

```
y = W2 ( SiLU(x·W_gate) ⊙ (x·W_up) )
```

## 特性

- 📦 **单文件，零配置**：复制 `swiglu.py`，然后 `from swiglu import SwiGLU`。
- 🚀 **gate 与 up 合并为一次矩阵乘法**：两个投影合并为 `w13` 一次 matmul。
- 🔥 **融合 Triton 内核**：`silu(gate) * up` 在一个内核中完成，直接读取打包后的张量（没有 `chunk` / `contiguous` 拷贝）。内部以 fp32 计算，按输入 dtype 写回。
- 🧠 **反向传播省显存**：只保存输入，反向内核直接把梯度写入打包布局。
- 🛟 **自动回退**：CPU、MPS、未安装 Triton、不支持的 dtype 或任何导入错误，都会使用纯 PyTorch 实现。
- 🦙 **兼容 LLaMA**：默认隐藏维度为 `int(8·dim/3)` 并向上取整到 `multiple_of` 的倍数，并提供加载 `w1 / w2 / w3` 权重的方法。

## 安装

无需通过 PyPI 安装，直接复制文件即可：

```bash
curl -O https://raw.githubusercontent.com/ENC3LL/SwiGLU/swiglu.py
```

环境要求：

- Python 3.9+
- PyTorch 2.x（必需）
- Triton（可选，用于启用融合内核）。在 Linux 上，PyTorch 的 CUDA 版本会自动把它作为依赖安装。在 Windows 上请使用与 PyTorch 版本匹配的社区版本，如 [`triton-windows`](https://github.com/triton-lang/triton-windows)。没有 Triton 时会自动使用 PyTorch 实现。

## 快速开始

### 作为一个层使用

```python
import torch
from swiglu import SwiGLU

ffn = SwiGLU(dim=4096).cuda().bfloat16()
x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
y = ffn(x)  # -> [2, 1024, 4096]
```

### 用在 Transformer 块中

```python
import torch.nn as nn
from swiglu import SwiGLU

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):
        return x + self.ffn(self.norm(x))
```

### 函数式接口

```python
from swiglu import swiglu, swiglu_packed

y = swiglu_packed(packed)   # packed: [..., 2H] -> [..., H]（CUDA 上使用融合 Triton 内核）
y = swiglu(gate, up)        # 两个张量 -> silu(gate) * up
```

### 加载 LLaMA 权重

```python
ffn = SwiGLU(dim=4096, hidden_dim=11008, multiple_of=1)
ffn.load_separate(w1=state["w1.weight"], w3=state["w3.weight"], w2=state["w2.weight"])
# w1 = gate，w3 = up，w2 = down
```

## API

| 对象 | 说明 |
|---|---|
| `SwiGLU(dim, hidden_dim=None, multiple_of=256, bias=False)` | 前馈模块。`hidden_dim` 为 `None` 时取 `int(8·dim/3)` 并向上取整到 `multiple_of` 的倍数。 |
| `SwiGLU.load_separate(w1, w3, w2)` | 把 LLaMA 风格的分离权重复制到合并后的 `w13` 和 `w2`。 |
| `swiglu_packed(x)` | `x: [..., 2H]` → `silu(x[..., :H]) * x[..., H:]`。在 CUDA 上使用 Triton（fp16 / bf16 / fp32）。 |
| `swiglu(gate, up)` | 对两个独立张量计算 `silu(gate) * up`。 |

## 自测

```bash
python swiglu.py
```

会输出与 fp32 参考实现相比的前向/反向最大误差，以及输出形状。

## 注意事项与限制

- Triton 路径：CUDA（以及 Triton 可用的 ROCm 版本），dtype 为 fp16 / bf16 / fp32。其余情况走 PyTorch。
- 融合路径不支持二阶导数（double backward）。
- `w13` 权重按 `[gate | up]` 排列，导出权重时请注意。

## 许可证

MIT。如使用其他许可证，请修改此处。

---

<sub>关键词：SwiGLU、PyTorch、Triton、LLaMA、Transformer、前馈网络、FFN、门控线性单元、GLU、SiLU、Swish、融合内核、大语言模型、CUDA、GPU、深度学习。</sub>
