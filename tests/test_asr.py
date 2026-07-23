"""ASR smoke tests (requires GPU)."""

from __future__ import annotations

from qwen_transkrib.asr import transcribe_file
from qwen_transkrib.config import Settings
from tests.conftest import requires_gpu


@requires_gpu
def test_transcribe_short_audio(short_audio_path):
    """Transcribe short audio file and verify output structure."""
    settings = Settings()
    words, text, language = transcribe_file(short_audio_path, settings)

    assert len(words) > 0, "Should produce at least one word"
    assert all(w.start < w.end for w in words), "All words should have valid timestamps"
    assert len(text) > 0, "Should produce non-empty text"
    assert language == "Russian", "Should detect Russian language"


@requires_gpu
def test_transcribe_with_context(short_audio_path):
    """Transcribe with context hotwords."""
    settings = Settings()
    words, text, language = transcribe_file(short_audio_path, settings, context="тест, проверка")

    assert len(words) > 0
    assert len(text) > 0
