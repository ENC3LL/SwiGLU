from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

__all__ = ["SwiGLU", "swiglu", "swiglu_packed"]

_BLOCK = 1024
_TRITON_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


if _HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(x_ptr, out_ptr, n, H, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        row = offs // H
        col = offs - row * H
        base = row * (2 * H) + col
        a = tl.load(x_ptr + base, mask=mask, other=0).to(tl.float32)
        b = tl.load(x_ptr + base + H, mask=mask, other=0).to(tl.float32)
        y = a * tl.sigmoid(a) * b
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(x_ptr, g_ptr, dx_ptr, n, H, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        row = offs // H
        col = offs - row * H
        base = row * (2 * H) + col
        a = tl.load(x_ptr + base, mask=mask, other=0).to(tl.float32)
        b = tl.load(x_ptr + base + H, mask=mask, other=0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0).to(tl.float32)
        s = tl.sigmoid(a)
        silu = a * s
        da = g * b * s * (1.0 + a * (1.0 - s))
        db = g * silu
        dt = dx_ptr.dtype.element_ty
        tl.store(dx_ptr + base, da.to(dt), mask=mask)
        tl.store(dx_ptr + base + H, db.to(dt), mask=mask)


class _SwiGLUPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        H = x.shape[-1] // 2
        out = torch.empty(*x.shape[:-1], H, device=x.device, dtype=x.dtype)
        n = out.numel()
        if n:
            with torch.cuda.device(x.device):
                _swiglu_fwd_kernel[(triton.cdiv(n, _BLOCK),)](x, out, n, H, BLOCK=_BLOCK)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        g = g.contiguous()
        H = x.shape[-1] // 2
        dx = torch.empty_like(x)
        n = g.numel()
        if n:
            with torch.cuda.device(x.device):
                _swiglu_bwd_kernel[(triton.cdiv(n, _BLOCK),)](x, g, dx, n, H, BLOCK=_BLOCK)
        return dx


def swiglu_packed(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("last dimension must be even (gate|up packed)")
    if _HAS_TRITON and x.is_cuda and x.dtype in _TRITON_DTYPES:
        return _SwiGLUPacked.apply(x)
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu_packed(self.w13(x)))

    @torch.no_grad()
    def load_separate(self, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> None:
        self.w13.weight.copy_(torch.cat([w1, w3], dim=0))
        self.w2.weight.copy_(w2)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    torch.manual_seed(0)

    x = torch.randn(4, 128, 1024, device=dev, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    out = swiglu_packed(x)
    gate, up = x_ref.float().chunk(2, dim=-1)
    ref = F.silu(gate) * up

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())

    fwd_err = (out.float() - ref).abs().max().item()
    bwd_err = (x.grad.float() - x_ref.grad.float()).abs().max().item()
    print(f"{dev} | forward max err: {fwd_err:.3e} | backward max err: {bwd_err:.3e}")

    model = SwiGLU(512).to(dev, dtype)
    y = model(torch.randn(4, 128, 512, device=dev, dtype=dtype))
    print("SwiGLU output shape:", tuple(y.shape))
