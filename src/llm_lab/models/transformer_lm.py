"""Compact transformer baseline for fair comparison with Mamba."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from llm_lab.models.layers import RMSNorm, init_weights


def _grad_checkpoint(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Checkpoint a layer; stock impl can't resolve the XLA device module, so use torch_xla's on TPU."""
    if x.device.type == "xla":
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
        return xla_checkpoint(layer, x)
    return torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)


def _xla_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal attention through torch_xla's Pallas flash kernel, (B, H, L, D) in and out.

    On XLA, SDPA lowers to the math path and materializes every head's L x L scores; the
    kernel tiles them in VMEM instead. Under SPMD the kernel needs the batch sharding spelled
    out (it is a custom call the partitioner can't see into), so pass the global mesh's batch
    axis. No attention dropout: the pretraining configs run dropout 0.
    """
    from torch_xla.experimental.custom_kernel import flash_attention
    import torch_xla.runtime as xr
    if xr.is_spmd():
        import torch_xla.distributed.spmd as xs
        mesh = xs.get_global_mesh()
        return flash_attention(q, k, v, causal=True, sm_scale=scale,
                               partition_spec=(mesh.axis_names[0], None, None, None), mesh=mesh)
    return flash_attention(q, k, v, causal=True, sm_scale=scale)


class TransformerLM(nn.Module):
    """Small causal transformer LM with RoPE."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        seq_len: int = 256,
        dropout: float = 0.0,
        use_grad_checkpoint: bool = False,
        tie_embeddings: bool = True,
        flash_attention: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, seq_len, dropout, flash_attention)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.use_grad_checkpoint = use_grad_checkpoint

        self.apply(init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            if self.use_grad_checkpoint and self.training:
                x = _grad_checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float,
                 flash_attention: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len, dropout, flash_attention)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int, dropout: float,
                 flash_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.flash_attention = flash_attention

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # precompute RoPE frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(seq_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer("cos", angles.cos().unsqueeze(0).unsqueeze(0))  # (1, 1, L, head_dim/2)
        self.register_buffer("sin", angles.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv.unbind(0)

        q = self._apply_rope(q, L)
        k = self._apply_rope(k, L)

        if self.flash_attention and x.device.type == "xla":
            attn = _xla_flash_attention(q, k, v, 1.0 / math.sqrt(self.head_dim))
        else:
            # scaled dot-product with causal mask
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        attn = attn.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(attn)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embedding."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = self.cos[:, :, :seq_len, :].to(x.dtype)
        sin = self.sin[:, :, :seq_len, :].to(x.dtype)
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)
