<div align="center">

# ⚡ SwiGLU для PyTorch

**Готовый однофайловый SwiGLU-слой: fused Triton-кернел и автоматический фолбэк на чистый PyTorch.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-optional-76B900?logo=nvidia&logoColor=white)](https://github.com/triton-lang/triton)
[![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Single file](https://img.shields.io/badge/single--file-drop--in-blue)](swiglu.py)

[English](README.md) · [简体中文](README.zh-CN.md) · **Русский**

</div>

---

## Что это?

Готовый блок **SwiGLU** (Swish-Gated Linear Unit), который используется в **LLaMA**, PaLM, Mistral и большинстве современных LLM. Скопируйте один файл в проект, импортируйте, и всё работает.

```
y = W2 ( SiLU(x·W_gate) ⊙ (x·W_up) )
```

## Возможности

- 📦 **Один файл, без настройки**: скопируйте `swiglu.py` и напишите `from swiglu import SwiGLU`.
- 🚀 **gate и up одним GEMM**: обе проекции объединены в один matmul `w13`.
- 🔥 **Fused Triton-кернелы**: `silu(gate) * up` считается одним кернелом, который читает packed-тензор напрямую (без копий `chunk` / `contiguous`). Вычисления в fp32, результат пишется в исходном dtype.
- 🧠 **Экономный backward**: сохраняется только вход, backward-кернел пишет градиент сразу в packed-раскладке.
- 🛟 **Автоматический фолбэк**: CPU, MPS, нет Triton, неподдерживаемый dtype или ошибка импорта → обычный PyTorch.
- 🦙 **Совместимость с LLaMA**: hidden по умолчанию `int(8·dim/3)`, округлённый вверх до `multiple_of`, и метод для загрузки весов `w1 / w2 / w3`.

## Установка

Через PyPI ставить нечего, просто скопируйте файл:

```bash
curl -O https://raw.githubusercontent.com/ENC3LL/SwiGLU/swiglu.py
```

Требования:

- Python 3.9+
- PyTorch 2.x (обязательно)
- Triton (опционально, включает fused-кернелы). На Linux CUDA-сборки PyTorch ставят его как зависимость. На Windows используйте community-сборку, например [`triton-windows`](https://github.com/triton-lang/triton-windows), версия которой соответствует вашему PyTorch. Без Triton код автоматически работает на PyTorch.

## Быстрый старт

### Как слой

```python
import torch
from swiglu import SwiGLU

ffn = SwiGLU(dim=4096).cuda().bfloat16()
x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)
y = ffn(x)  # -> [2, 1024, 4096]
```

### В блоке Transformer

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

### Функциональный API

```python
from swiglu import swiglu, swiglu_packed

y = swiglu_packed(packed)   # packed: [..., 2H] -> [..., H]  (fused Triton на CUDA)
y = swiglu(gate, up)        # два тензора -> silu(gate) * up
```

### Загрузка весов LLaMA

```python
ffn = SwiGLU(dim=4096, hidden_dim=11008, multiple_of=1)
ffn.load_separate(w1=state["w1.weight"], w3=state["w3.weight"], w2=state["w2.weight"])
# w1 = gate, w3 = up, w2 = down
```

## API

| Объект | Описание |
|---|---|
| `SwiGLU(dim, hidden_dim=None, multiple_of=256, bias=False)` | Feed-forward модуль. Если `hidden_dim=None`, берётся `int(8·dim/3)`, округлённое вверх до кратного `multiple_of`. |
| `SwiGLU.load_separate(w1, w3, w2)` | Копирует раздельные веса в стиле LLaMA в объединённые `w13` и `w2`. |
| `swiglu_packed(x)` | `x: [..., 2H]` → `silu(x[..., :H]) * x[..., H:]`. На CUDA использует Triton (fp16 / bf16 / fp32). |
| `swiglu(gate, up)` | `silu(gate) * up` для двух отдельных тензоров. |

## Самопроверка

```bash
python swiglu.py
```

Выводит максимальную ошибку forward/backward относительно fp32-эталона и форму выхода.

## Замечания и ограничения

- Triton-путь: CUDA (и ROCm-сборки, где доступен Triton), dtype fp16 / bf16 / fp32. Всё остальное идёт через PyTorch.
- Double backward (производные второго порядка) fused-путь не поддерживает.
- Вес `w13` хранится как `[gate | up]`, учитывайте это при экспорте чекпоинтов.

## Лицензия

MIT. Измените этот раздел, если используете другую лицензию.

---

<sub>Ключевые слова: SwiGLU, PyTorch, Triton, LLaMA, Transformer, feed-forward network, FFN, gated linear unit, GLU, SiLU, Swish, fused kernel, LLM, CUDA, GPU, deep learning, нейросети, глубокое обучение.</sub>
