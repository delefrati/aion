"""Model factory — no heavy dependencies (no numpy, no datasets)."""
from __future__ import annotations

from llm_lab.training.config import TrainConfig


def build_model(cfg: TrainConfig):
    """Build model based on config model_type."""
    if cfg.model.model_type == "mamba":
        from llm_lab.models.mamba_lm import MambaLM
        return MambaLM(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            d_state=cfg.model.d_state,
            d_conv=cfg.model.d_conv,
            expand=cfg.model.expand,
            use_grad_checkpoint=cfg.device.grad_checkpoint,
        )
    elif cfg.model.model_type == "transformer":
        from llm_lab.models.transformer_lm import TransformerLM
        return TransformerLM(
            vocab_size=cfg.model.vocab_size,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            seq_len=cfg.model.seq_len,
            use_grad_checkpoint=cfg.device.grad_checkpoint,
            tie_embeddings=cfg.model.tie_embeddings,
            dropout=cfg.model.dropout,
            flash_attention=cfg.device.flash_attention,
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model.model_type}")
