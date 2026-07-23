"""Diarization smoke tests (requires GPU + HF_TOKEN)."""

from __future__ import annotations

import os

import pytest

from qwen_transkrib.config import Settings
from qwen_transkrib.diarize import diarize_file
from tests.conftest import requires_gpu


@requires_gpu
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN required for pyannote models",
)
def test_diarize_short_audio(short_audio_path):
    """Diarize short audio file and verify output structure."""
    settings = Settings()
    df = diarize_file(short_audio_path, settings)

    assert not df.empty, "Should produce diarization turns"
    assert "start" in df.columns
    assert "end" in df.columns
    assert "speaker" in df.columns
    assert all(df["start"] < df["end"]), "All turns should have valid timestamps"
