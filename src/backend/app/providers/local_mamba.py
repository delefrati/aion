"""Local Mamba model provider for the AION backend.

Loads a fine-tuned Mamba checkpoint and serves generation requests.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

from app.providers.base import BaseProvider

logger = logging.getLogger("aion.mamba")

# Add llm_lab to path if not installed as package
_lab_path = Path(__file__).resolve().parent.parent.parent / "llm_lab"
if _lab_path.exists() and str(_lab_path.parent) not in sys.path:
    sys.path.insert(0, str(_lab_path.parent))


class LocalMambaProvider(BaseProvider):
    """Serves generation from a local fine-tuned Mamba model."""

    def __init__(self, model_path: str, tokenizer_path: str, config_path: str | None = None, device: str = "auto"):
        from llm_lab.training.config import TrainConfig
        from llm_lab.training.model_factory import build_model
        from llm_lab.tokenizer.bpe import load_tokenizer
        from llm_lab.utils import pick_device
        import yaml

        resolved_model_path = self._resolve_checkpoint_path(model_path)
        bundle = torch.load(resolved_model_path, map_location="cpu", weights_only=False)

        if "config" in bundle:
            cfg_dict = bundle["config"]
        else:
            # Checkpoint predates config saving — load from yaml file
            if config_path is None:
                raise ValueError("Checkpoint has no embedded config; set AION_MAMBA_CONFIG_PATH")
            cfg_dict = yaml.safe_load(Path(config_path).read_text())

        cfg = TrainConfig.from_dict(cfg_dict)

        self.device = pick_device(device)

        self.model = build_model(cfg).to(self.device)
        self.model.load_state_dict(bundle["model"])
        self.model.eval()

        self.tokenizer = load_tokenizer(Path(tokenizer_path))

        from app.config import settings
        self.max_tokens = settings.max_tokens_per_request
        # Trained context window; total (prompt + generated) must stay within this
        self.context_len = getattr(cfg, "seq_len", self.max_tokens)
        self.history_turns = settings.mamba_history_turns
        self.temperature = 0.8   # conversational temperature; 0.3 was too deterministic

    @staticmethod
    def _resolve_checkpoint_path(model_path: str) -> str:
        """Prefer best.pt, but gracefully fall back to latest.pt when needed."""
        path = Path(model_path)
        if path.exists():
            return str(path)

        # If configured path is best.pt and it's missing, try latest.pt in same folder.
        if path.name == "best.pt":
            fallback = path.with_name("latest.pt")
            if fallback.exists():
                return str(fallback)

        raise FileNotFoundError(f"Checkpoint not found: {path}")

    async def generate(self, message: str, history: list[dict], params: dict | None = None) -> str:
        params = params or {}
        prompt = self._format_prompt(message, history, params.get("history_turns"))
        output = self._generate_sync(prompt, params)
        return self._extract_response(output)

    def _format_prompt(self, message: str, history: list[dict], history_turns: int | None = None) -> str:
        """Format as chat prompt using the multi-turn template.

        Trims old turns so the prompt leaves at least half the context window
        for generation.
        """
        max_prompt_tokens = self.context_len // 2
        turns = self.history_turns if history_turns is None else history_turns

        # Current user turn is always included
        suffix = f"<|user|>{message}<|end|>\n<|assistant|>"
        suffix_len = len(self.tokenizer.encode(suffix).ids)

        # Add history turns newest-first until we'd exceed the prompt budget.
        # Keep it short: a weak model conditions on (and echoes) its own prior
        # answers, so a long history feeds a self-reinforcing repetition loop.
        kept: list[str] = []
        used = suffix_len
        for m in reversed(history[-turns:] if turns else []):
            role_tag = "<|user|>" if m["role"] == "user" else "<|assistant|>"
            turn = f"{role_tag}{m['content']}<|end|>"
            turn_len = len(self.tokenizer.encode(turn).ids)
            if used + turn_len > max_prompt_tokens:
                break
            kept.append(turn)
            used += turn_len

        kept.reverse()
        kept.append(suffix)
        return "\n".join(kept)

    @torch.no_grad()
    def _generate_sync(self, prompt: str, params: dict | None = None) -> str:
        """Run autoregressive generation, reusing the lab's sampler."""
        from llm_lab.eval.metrics import generate

        params = params or {}
        prompt_len = len(self.tokenizer.encode(prompt).ids)
        # Cap generation so prompt + generated stays within the trained context window
        requested_tokens = params.get("max_tokens", self.max_tokens)
        budget = max(1, min(requested_tokens, self.context_len - prompt_len))
        end_marker = self.tokenizer.encode("<|end|>").ids
        logger.info("Generating: prompt=%d tokens, budget=%d tokens", prompt_len, budget)
        t0 = time.monotonic()

        output = generate(
            self.model, self.tokenizer, prompt,
            max_tokens=budget, temperature=params.get("temperature", self.temperature),
            top_k=params.get("top_k", 40), top_p=params.get("top_p", 0.9),
            repetition_penalty=params.get("repetition_penalty", 1.3),
            stop_sequences=[end_marker] if end_marker else None,
        )
        logger.info("Generated in %.1fs", time.monotonic() - t0)
        return output

    def _extract_response(self, full_output: str) -> str:
        """Extract the last assistant response from generated text."""
        marker = "<|assistant|>"
        if marker in full_output:
            response = full_output.split(marker)[-1]
        else:
            # Fallback for old Q:/A: format checkpoints
            qa_marker = "\nA: "
            response = full_output.split(qa_marker)[-1] if qa_marker in full_output else full_output

        # Strip end marker
        end = "<|end|>"
        hit_end_marker = end in response
        if hit_end_marker:
            response = response[:response.index(end)]
        else:
            # Generation was cut by max_tokens — trim to last complete sentence
            for punct in (".", "!", "?"):
                last = response.rfind(punct)
                if last != -1:
                    response = response[: last + 1]
                    break

        return response.strip() or self.FALLBACK
