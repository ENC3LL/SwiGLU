<div align="center">

# ⚡ SwiGLU for PyTorch

**Drop-in, single-file SwiGLU feed-forward layer with a fused Triton kernel and automatic PyTorch fallback.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-optional-76B900?logo=nvidia&logoColor=white)](https://github.com/triton-lang/triton)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Single file](https://img.shields.io/badge/single--file-drop--in-blue)](swiglu.py)

**English** · [简体中文](README.zh-CN.md) · [Русский](README.ru.md)

</div>

---

## What is this?

A ready-to-use **SwiGLU** (Swish-Gated Linear Unit) feed-forward block as used in **LLaMA**, PaLM, Mistral and most modern LLMs. Copy one file into your project, import it, done.

```
y = W2 ( SiLU(x·W_gate) ⊙ (x·W_up) )
```

## Features

- 📦 **One file, zero setup**: copy `swiglu.py`, then `from swiglu import SwiGLU`.
- 🚀 **Single GEMM for gate and up**: the two projections are fused into one `w13` matmul.
- 🔥 **Fused Triton kernels**: `silu(gate) * up` runs in one kernel that reads the packed tensor directly (no `chunk` / `contiguous` copies). Math is done in fp32 and the result is written in the input dtype.
- 🧠 **Memory-friendly backward**: only the input is saved, the backward kernel writes the gradient straight into the packed layout.
- 🛟 **Automatic fallback**: CPU, MPS, missing Triton, unsupported dtype, or any import error → plain PyTorch.
- 🦙 **LLaMA-compatible**: default hidden size `int(8·dim/3)` rounded up to `multiple_of`, and a helper to load `w1 / w2 / w3` checkpoints.

## Installation

Nothing to install from PyPI. Just copy the file:

```bash
curl -O https://raw.githubusercontent.com/ENC3LL/SwiGLU/swiglu.py
```

Requirements:

- Python 3.9+
- PyTorch 2.x (required)
- Triton (optional, enables the fused kernels). On Linux, CUDA builds of PyTorch pull it in as a dependency. On Windows use a community build such as [`triton-windows`](https://github.com/triton-lang/triton-windows) whose version matches your PyTorch. Without Triton, the code silently uses the PyTorch path.

## Quick start

### As a layer

```python
import torch
from swiglu import SwiGLU

ffn = SwiGLU(dim=4096).cuda().bfloat16()
x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
y = ffn(x)  # -> [2, 1024, 4096]
```

### Inside a Transformer block

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

### Functional API

```python
from swiglu import swiglu, swiglu_packed

y = swiglu_packed(packed)   # packed: [..., 2H] -> [..., H]  (fused Triton on CUDA)
y = swiglu(gate, up)        # two tensors -> silu(gate) * up
```

### Loading LLaMA weights

```python
ffn = SwiGLU(dim=4096, hidden_dim=11008, multiple_of=1)
ffn.load_separate(w1=state["w1.weight"], w3=state["w3.weight"], w2=state["w2.weight"])
# w1 = gate, w3 = up, w2 = down
```

## API

| Object | Description |
|---|---|
| `SwiGLU(dim, hidden_dim=None, multiple_of=256, bias=False)` | Feed-forward module. If `hidden_dim` is `None` it becomes `int(8·dim/3)` rounded up to a multiple of `multiple_of`. |
| `SwiGLU.load_separate(w1, w3, w2)` | Copy LLaMA-style separate weights into the packed `w13` + `w2`. |
| `swiglu_packed(x)` | `x: [..., 2H]` → `silu(x[..., :H]) * x[..., H:]`. Uses Triton on CUDA (fp16 / bf16 / fp32). |
| `swiglu(gate, up)` | `silu(gate) * up` for two separate tensors. |

## Self-test

```bash
python swiglu.py
```

Prints forward/backward max error against an fp32 reference and the output shape.

## Notes and limitations

- Triton path: CUDA (and ROCm builds where Triton is available), dtypes fp16 / bf16 / fp32. Everything else goes through PyTorch.
- Double backward (second-order gradients) is not supported by the fused path.
- The `w13` weight is stored as `[gate | up]`. Keep this in mind when exporting checkpoints.

## License

MIT.

---

<sub>Keywords: SwiGLU, PyTorch, Triton, LLaMA, Transformer, feed-forward network, FFN, gated linear unit, GLU, SiLU, Swish, fused kernel, LLM, CUDA, GPU, deep learning.</sub>
