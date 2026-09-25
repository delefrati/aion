"""Central per-run training budgets so the notebooks stay thin.

Notebooks import this AFTER cloning the repo (Cell 1 puts src/ on sys.path), so
tuning a run's step budget is a repo edit + `git push` — no notebook re-upload.
STEPS_PER_SESSION stays in the notebooks (it maps to the platform time limit, not
the model); only the total target, which varies per run and caused the chat
overfitting overshoot, lives here.
"""

# name -> total target steps (the hard cap the notebook stops at)
BUDGETS = {
    "pretrain_large": 72000,       # 235M base: compute-optimal ~4.7B tokens (needs pretrain_xl data to avoid repeats)
    "pretrain_large_edu": 20000,   # 235M continued pretraining on pretrain_edu (~1.3B of its ~2B tokens)
    "pretrain_medium": 20000,      # 110M base pretrain
    "pretrain_tpu_medium": 40000,  # 110M TPU continue (20k->40k warm restart)
    # 235M chat finetune. The old 8000 cap was the knee for a ~94k-example corpus (~2.6k
    # steps/epoch at eff batch 32): val bottomed at 1.766 @7.5k = ~3 epochs, then climbed.
    # The corpus is now ~225k examples (~7k steps/epoch), so the same ~3 epochs lands near
    # 20k. Treat this as a ceiling, not a target — re-read the val curve after the first two
    # sessions and pull it in if the sawtooth minima start rising again.
    "chat_large": 20000,
    "chat_medium": 20000,          # 110M chat finetune
    "chat_multisession": 20000,    # generic multi-session chat finetune
}


def budget(name: str) -> int:
    """Total target steps for a named run. Raises if the name is unknown."""
    if name not in BUDGETS:
        raise KeyError(f"Unknown run budget '{name}'. Known: {sorted(BUDGETS)}")
    return BUDGETS[name]


# name -> raw-corpus download preset (see data/download.py PRESETS). The 235M base uses the
# enlarged pretrain_xl (~5B tokens) so its extended 72k-step run sees new data, not repeats.
DATA_PRESETS = {
    "pretrain_large": "pretrain_xl",
    "pretrain_large_edu": "pretrain_edu",  # built off-device by tools/build_edu_cache.py
    "pretrain_medium": "pretrain",
    "pretrain_tpu_medium": "pretrain",
}


def data_preset(name: str) -> str:
    """Raw-corpus preset for a named run; defaults to the standard 'pretrain'."""
    return DATA_PRESETS.get(name, "pretrain")
