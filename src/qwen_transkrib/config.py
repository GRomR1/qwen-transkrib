"""Configuration settings via pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables with QWEN_ prefix."""

    model_config = {"env_prefix": "QWEN_"}

    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    language: str = "Russian"
    device: str = "cuda:0"
    max_new_tokens: int = 1024
    batch_size: int = 8
    gap_threshold: float = 2.0
