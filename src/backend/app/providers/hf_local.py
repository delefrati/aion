"""HuggingFace local model provider for the AION backend.

Loads any causal LM from HuggingFace (e.g. SmolLM2, Mamba, Qwen2.5)
and serves generation requests via the standard Provider contract.

Recommended small models for WL-1 hardware (8 GB RAM, CPU-only):
  - HuggingFaceTB/SmolLM2-360M-Instruct   (~1.4 GB)
  - HuggingFaceTB/SmolLM2-1.7B-Instruct   (~7 GB, tight)
  - Qwen/Qwen2.5-0.5B-Instruct            (~2 GB)
  - state-spaces/mamba-370m               (~1.5 GB, no chat template)

Set via env: AION_HF_MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from app.providers.base import BaseProvider


class HFLocalProvider(BaseProvider):
    """Serves generation from any HuggingFace CausalLM checkpoint."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        max_new_tokens: int = 256,
        torch_dtype: str = "auto",
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        if device == "auto":
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _device = device

        # dtype: keep float32 on CPU to avoid bfloat16 ops failing on old CPUs
        if torch_dtype == "auto":
            import torch
            _dtype = torch.float16 if _device == "cuda" else torch.float32
        else:
            import torch
            _dtype = getattr(torch, torch_dtype)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_dtype,
            device_map=_device,
        )
        self._model.eval()
        self._device = _device
        self._max_new_tokens = max_new_tokens

        # build a pipeline so we get chat-template formatting for free when supported
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device_map=_device,
        )

    # ------------------------------------------------------------------ #
    # Provider contract                                                    #
    # ------------------------------------------------------------------ #

    async def generate(self, message: str, history: list[dict]) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, message, history)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, message: str, history: list[dict]) -> str | list[dict]:
        """Return a chat-formatted prompt if the tokenizer supports it, else plain text."""
        if self._tokenizer.chat_template is not None:
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in history[-6:]  # keep last 3 turns
            ]
            messages.append({"role": "user", "content": message})
            return messages  # pipeline handles apply_chat_template
        # Fallback for models without a chat template (e.g. base mamba checkpoints)
        context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
        return f"{context}\nuser: {message}\nassistant:"

    def _generate_sync(self, message: str, history: list[dict]) -> str:
        prompt = self._build_prompt(message, history)

        result = self._pipe(
            prompt,
            max_new_tokens=self._max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,
        )

        # pipeline returns a list; index depends on whether input was str or messages
        output = result[0]
        if isinstance(output, dict):
            text = output.get("generated_text", "")
        elif isinstance(output, list):
            # chat-messages output: last entry is the assistant turn
            text = output[-1].get("content", "") if output else ""
        else:
            text = str(output)

        return text.strip() or self.FALLBACK
