from __future__ import annotations

from enum import Enum

from pydantic_settings import BaseSettings


class AionMode(str, Enum):
    nano = "nano"
    standard = "standard"
    lab = "lab"


class AionProvider(str, Enum):
    deterministic = "deterministic"
    local_mamba = "local-mamba"
    hf_local = "hf-local"


class Settings(BaseSettings):
    mode: AionMode = AionMode.nano
    provider: AionProvider = AionProvider.deterministic

    host: str = "0.0.0.0"
    port: int = 8900

    db_path: str = "data/aion.db"
    database_url: str = "postgresql://aion:aion@postgres:5432/aion"

    max_tokens_per_request: int = 128

    # Local model serving paths (lab-data volume mounted at /app/lab-data)
    mamba_model_path: str = "/app/lab-data/checkpoints/mamba_instruct/model_serving.pt"
    mamba_tokenizer_path: str = "/app/lab-data/tokenizer.json"
    mamba_config_path: str | None = None  # Required if checkpoint has no embedded config
    # How many prior chat turns to feed back into the prompt. Keep small for weak
    # models: long history feeds a self-reinforcing repetition loop.
    mamba_history_turns: int = 2
    model_device: str = "auto"

    # HuggingFace local provider
    # Examples: HuggingFaceTB/SmolLM2-360M-Instruct, Qwen/Qwen2.5-0.5B-Instruct
    hf_model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    hf_max_new_tokens: int = 256
    hf_torch_dtype: str = "auto"  # auto | float32 | float16 | bfloat16

    model_config = {"env_prefix": "AION_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
