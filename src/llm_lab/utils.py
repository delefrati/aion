"""Small shared utilities."""
from __future__ import annotations

import torch


def pick_device(device: str = "auto") -> torch.device:
    """Resolve a serving/CLI device string.

    'auto' selects CUDA when available, else CPU. Training uses
    trainer._get_device() which also considers TPU/XLA; serving and the CLI
    never run on XLA, so this stays simple.
    """
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
