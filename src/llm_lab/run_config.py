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
    "pretrain_medium": 20000,      # 110M base pretrain
    "pretrain_tpu_medium": 40000,  # 110M TPU continue (20k->40k warm restart)
    "chat_large": 8000,            # 235M chat finetune — overfits past ~7.5k, cap at the knee
    "chat_medium": 20000,          # 110M chat finetune
    "chat_multisession": 20000,    # generic multi-session chat finetune
}


def budget(name: str) -> int:
    """Total target steps for a named run. Raises if the name is unknown."""
    if name not in BUDGETS:
        raise KeyError(f"Unknown run budget '{name}'. Known: {sorted(BUDGETS)}")
    return BUDGETS[name]
