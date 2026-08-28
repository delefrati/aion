"""Mamba-based language model (from scratch).

This implements the core Mamba selective state-space block
and wraps it into a causal LM.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_lab.models.layers import RMSNorm, init_weights


class _SelectiveScanFn(torch.autograd.Function):
    """Custom autograd for selective scan — Triton forward + Triton backward."""

    @staticmethod
    def forward(ctx, x, dt, A, B, C):
        """Run Triton forward, store h_states for backward."""
        from llm_lab.models.triton_scan import triton_selective_scan_fwd
        y, h_states = triton_selective_scan_fwd(x, dt, A, B, C)
        ctx.save_for_backward(x, dt, A, B, C, h_states)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """Run Triton backward kernel."""
        from llm_lab.models.triton_scan import triton_selective_scan_bwd
        x, dt, A, B, C, h_states = ctx.saved_tensors
        return triton_selective_scan_bwd(x, dt, A, B, C, h_states, grad_y)


def _sequential_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Sequential selective scan — differentiable, for backward recomputation."""
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]

    A_expanded = A.unsqueeze(0)  # (1, D, N)
    h = x.new_zeros(batch, d_inner, d_state)
    y = torch.empty(batch, seq_len, d_inner, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        dt_t = dt[:, t, :, None]  # (B, D, 1)
        dA = torch.exp(A_expanded * dt_t)  # (B, D, N)
        dB = dt_t * B[:, t, None, :]  # (B, D, N)
        h = dA * h + dB * x[:, t, :, None]  # (B, D, N)
        y[:, t] = torch.einsum('bdn,bn->bd', h, C[:, t])

    return y


class MambaBlock(nn.Module):
    """Single Mamba block with selective scan."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        # input projection: x -> (z, x_proj) where z is the gate
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1-d depthwise conv over sequence
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # SSM parameters projected from x
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, dt

        # dt projection
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # A is log-parameterized for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.d_inner = d_inner
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape

        # project and split into x and gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_part, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # conv over sequence dim
        x_conv = x_part.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: trim to L
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        # SSM parameters from x
        ssm_params = self.x_proj(x_conv)  # (B, L, d_state*2 + 1)
        B_param = ssm_params[:, :, :self.d_state]
        C_param = ssm_params[:, :, self.d_state:2*self.d_state]
        dt = ssm_params[:, :, -1:]  # (B, L, 1)

        # dt projection and softplus for positivity
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability

        # selective scan
        y = self._selective_scan(x_conv, dt, A, B_param, C_param)

        # gate and project out
        y = y * F.silu(z)
        y = y + x_part * self.D.unsqueeze(0).unsqueeze(0)  # skip connection
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C):
        """Route to best available scan implementation.

        Training on CUDA: custom autograd (Triton fwd, sequential bwd with recompute)
        Inference on CUDA: Triton (no backward needed)
        CPU fallback: sequential scan
        """
        # Debug switch to bypass Triton and isolate host-RAM leaks (AION_FORCE_SEQ_SCAN=1).
        if x.is_cuda and not os.environ.get("AION_FORCE_SEQ_SCAN"):
            if self.training:
                return _SelectiveScanFn.apply(x, dt, A, B, C)
            else:
                from llm_lab.models.triton_scan import triton_selective_scan
                return triton_selective_scan(x, dt, A, B, C)

        return _sequential_scan(x, dt, A, B, C)


class MambaLM(nn.Module):
    """Mamba language model: embedding -> N x (MambaBlock + RMSNorm) -> LM head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": MambaBlock(d_model, d_state, d_conv, expand),
                "norm": RMSNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.use_grad_checkpoint = use_grad_checkpoint

        # weight tying
        self.lm_head.weight = self.embedding.weight

        self.apply(init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, L) -> logits: (B, L, vocab_size)"""
        x = self.embedding(input_ids)
        for layer in self.layers:
            if self.use_grad_checkpoint and self.training:
                x = x + torch.utils.checkpoint.checkpoint(
                    self._layer_forward, layer, x, use_reentrant=False
                )
            else:
                x = x + layer["mamba"](layer["norm"](x))
        x = self.norm_f(x)
        return self.lm_head(x)

    @staticmethod
    def _layer_forward(layer, x):
        return layer["mamba"](layer["norm"](x))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
