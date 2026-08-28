"""Model factory — no heavy dependencies (no numpy, no datasets)."""
from __future__ import annotations

from llm_lab.training.config import TrainConfig


def build_model(cfg: TrainConfig):
    """Build model based on config model_type."""
    if cfg.model_type == "mamba":
        from llm_lab.models.mamba_lm import MambaLM
        return MambaLM(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            use_grad_checkpoint=cfg.grad_checkpoint,
        )
    elif cfg.model_type == "transformer":
        from llm_lab.models.transformer_lm import TransformerLM
        return TransformerLM(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            seq_len=cfg.seq_len,
            use_grad_checkpoint=cfg.grad_checkpoint,
            tie_embeddings=getattr(cfg, "tie_embeddings", True),
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")
