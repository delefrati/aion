#!/usr/bin/env python3
"""
Test and compare old vs new trained checkpoints locally.

Usage:
    python test_checkpoint_improvements.py \\
      --old training-data/latest.pt \\
      --new /path/to/step_1000.pt \\
      --config /path/to/run.yaml
"""
import argparse
import sys
from pathlib import Path

import torch


def load_model_and_tokenizer(checkpoint_path: str, config_path: str | None = None):
    """Load a trained model, tokenizer, and config."""
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from llm_lab.training.config import TrainConfig
    from llm_lab.training.model_factory import build_model
    from llm_lab.tokenizer.bpe import load_tokenizer
    import yaml

    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" in bundle:
        cfg_dict = bundle["config"]
    else:
        if config_path is None:
            raise ValueError(f"Checkpoint {checkpoint_path} has no embedded config; provide --config")
        cfg_dict = yaml.safe_load(Path(config_path).read_text())

    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in TrainConfig.__dataclass_fields__})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(bundle["model"])
    model.eval()

    # Tokenizer path from config
    tokenizer_path = Path(cfg_dict.get("tokenizer_path", "training-data/tokenizer.json"))
    if not tokenizer_path.exists():
        # Try relative to checkpoint dir
        tokenizer_path = Path(checkpoint_path).parent / "tokenizer.json"
    
    tokenizer = load_tokenizer(tokenizer_path)

    return model, tokenizer, device, cfg


def generate(model, tokenizer, prompt: str, device, max_tokens: int = 100, temperature: float = 0.8):
    """Generate text from a prompt."""
    encoded = tokenizer.encode(prompt)
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)
            logits = logits[:, -1, :] / temperature

            # top-k sampling
            top_k = 20
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)

            # Stop at end markers
            if tokenizer.token_to_id("<eos>") and next_id.item() == tokenizer.token_to_id("<eos>"):
                break
            if tokenizer.encode("<|end|>").ids and ids[0, -1:].item() in tokenizer.encode("<|end|>").ids:
                break

    return tokenizer.decode(ids[0].tolist())


def extract_response(text: str) -> str:
    """Extract assistant response from multi-turn format."""
    marker = "<|assistant|>"
    if marker in text:
        response = text.split(marker)[-1]
    else:
        # Fallback for old Q:/A: format
        qa_marker = "\nA: "
        response = text.split(qa_marker)[-1] if qa_marker in text else text

    end = "<|end|>"
    if end in response:
        response = response[:response.index(end)]

    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Test checkpoint improvements")
    parser.add_argument("--old", required=True, help="Path to old checkpoint")
    parser.add_argument("--new", required=True, help="Path to new checkpoint")
    parser.add_argument("--config", help="Path to training config YAML (if not embedded)")
    args = parser.parse_args()

    # Test prompts
    prompts = [
        "<|user|>Hello, how are you?<|end|>\n<|assistant|>",
        "<|user|>What is machine learning?<|end|>\n<|assistant|>",
        "<|user|>Tell me a joke.<|end|>\n<|assistant|>",
    ]

    print("\n" + "=" * 80)
    print("CHECKPOINT COMPARISON TEST")
    print("=" * 80)

    # Load models
    print("\nLoading old model...", end=" ", flush=True)
    old_model, old_tok, old_dev, old_cfg = load_model_and_tokenizer(args.old, args.config)
    print(f"✓ ({old_cfg.model_type}, {sum(p.numel() for p in old_model.parameters()):,} params)")

    print("Loading new model...", end=" ", flush=True)
    new_model, new_tok, new_dev, new_cfg = load_model_and_tokenizer(args.new, args.config)
    print(f"✓ ({new_cfg.model_type}, {sum(p.numel() for p in new_model.parameters()):,} params)")

    # Run tests
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'-' * 80}")
        print(f"Test {i}: {prompt.split('|')[1].strip()[:50]}...")
        print(f"{'-' * 80}")

        print("\n[OLD MODEL]")
        old_output = generate(old_model, old_tok, prompt, old_dev, max_tokens=80, temperature=0.8)
        old_response = extract_response(old_output)
        print(f"Response: {old_response[:200]}...")

        print("\n[NEW MODEL]")
        new_output = generate(new_model, new_tok, prompt, new_dev, max_tokens=80, temperature=0.8)
        new_response = extract_response(new_output)
        print(f"Response: {new_response[:200]}...")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
