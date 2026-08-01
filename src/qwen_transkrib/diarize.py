"""Speaker diarization module wrapping pyannote."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as _ta_func
from pyannote.audio import Pipeline

# Monkey-patch: torchaudio is a stub on MetaX GPUs, so we provide a
# working resample via linear interpolation.
if not hasattr(_ta_func, "resample") or not callable(getattr(_ta_func, "resample", None)):

    def _resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
        if orig_freq == new_freq:
            return waveform
        new_length = int(waveform.shape[-1] * new_freq / orig_freq)
        # interpolate expects (batch, length) or (batch, channels, length)
        unsqueezed = waveform.unsqueeze(0) if waveform.dim() < 3 else waveform
        resampled = torch.nn.functional.interpolate(
            unsqueezed.float(),
            size=new_length,
            mode="linear",
            align_corners=False,
        )
        return resampled.squeeze(0) if waveform.dim() < 3 else resampled

    _ta_func.resample = _resample  # type: ignore[attr-defined]

from qwen_transkrib.config import Settings
from qwen_transkrib.errors import DiarizationError

_pipeline_cache: dict[str, Pipeline] = {}


def _get_pipeline(settings: Settings) -> Pipeline:
    """Get or create cached diarization pipeline."""
    key = f"{settings.diarization_model}:{settings.device}"
    if key not in _pipeline_cache:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise DiarizationError(
                "HF_TOKEN environment variable is required for pyannote models. "
                "Get yours at: https://huggingface.co/settings/tokens"
            )
        try:
            pipe = Pipeline.from_pretrained(
                settings.diarization_model,
                token=token,
            )
            pipe.to(torch.device(settings.device))
            _pipeline_cache[key] = pipe
        except Exception as e:
            raise DiarizationError(
                f"Failed to load diarization model '{settings.diarization_model}': {e}"
            ) from e
    return _pipeline_cache[key]


def diarize_file(
    path: Path,
    settings: Settings,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> pd.DataFrame:
    """Run speaker diarization on audio file.

    Args:
        path: Path to audio file.
        settings: Application settings.
        min_speakers: Minimum number of speakers. Defaults to 2 if not set.
        max_speakers: Maximum number of speakers. Defaults to 2 if not set.

    Returns:
        DataFrame with columns: start, end, speaker.
    """
    pipeline = _get_pipeline(settings)

    # Load audio as tensor (avoids torchcodec issues on MetaX)
    waveform, sr = sf.read(str(path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    audio_input = {
        "waveform": torch.from_numpy(waveform).unsqueeze(0),
        "sample_rate": sr,
    }

    # Default to 2 speakers for most conversation audio
    kwargs: dict = {
        "min_speakers": min_speakers if min_speakers is not None else 2,
        "max_speakers": max_speakers if max_speakers is not None else 2,
    }

    try:
        diarization = pipeline(audio_input, **kwargs)
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e

    # pyannote 4.0 returns DiarizeOutput; extract the Annotation object
    annotation = getattr(diarization, "speaker_diarization", diarization)

    rows = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    return pd.DataFrame(rows)


def clear_cache() -> None:
    """Clear cached pipelines to free memory."""
    _pipeline_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
