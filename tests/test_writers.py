"""Unit tests for output format writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_transkrib.schemas import Segment, TranscriptionResult, Word
from qwen_transkrib.writers import get_writer
from qwen_transkrib.writers.json import write_json
from qwen_transkrib.writers.srt import write_srt
from qwen_transkrib.writers.txt import write_txt
from qwen_transkrib.writers.vtt import write_vtt


def _make_result() -> TranscriptionResult:
    """Create a sample TranscriptionResult for testing."""
    return TranscriptionResult(
        audio_path="/path/to/audio.wav",
        duration_sec=10.5,
        language="Russian",
        model="Qwen/Qwen3-ASR-1.7B",
        diarization_model="pyannote/speaker-diarization-community-1",
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            Segment(
                start=0.0,
                end=5.0,
                speaker="SPEAKER_00",
                text="Привет мир",
                words=[
                    Word(text="Привет", start=0.0, end=2.5),
                    Word(text="мир", start=2.6, end=5.0),
                ],
            ),
            Segment(
                start=5.5,
                end=10.0,
                speaker="SPEAKER_01",
                text="Как дела",
                words=[
                    Word(text="Как", start=5.5, end=7.0),
                    Word(text="дела", start=7.1, end=10.0),
                ],
            ),
        ],
        full_text="Привет мир Как дела",
    )


def test_txt_format(tmp_path: Path):
    """TXT writer produces correct format."""
    result = _make_result()
    out = tmp_path / "test.txt"

    write_txt(result, out)

    content = out.read_text()
    assert "[SPEAKER_00] Привет мир" in content
    assert "[SPEAKER_01] Как дела" in content


def test_srt_format(tmp_path: Path):
    """SRT writer produces valid SRT format."""
    result = _make_result()
    out = tmp_path / "test.srt"

    write_srt(result, out)

    content = out.read_text()
    # Check SRT structure: number, timestamp, text
    assert "1\n" in content
    assert "00:00:00,000 --> 00:00:05,000" in content
    assert "[SPEAKER_00] Привет мир" in content
    assert "2\n" in content
    assert "00:00:05,500 --> 00:00:10,000" in content
    assert "[SPEAKER_01] Как дела" in content


def test_vtt_format(tmp_path: Path):
    """VTT writer produces valid WebVTT format."""
    result = _make_result()
    out = tmp_path / "test.vtt"

    write_vtt(result, out)

    content = out.read_text()
    assert content.startswith("WEBVTT\n\n")
    assert "<v SPEAKER_00>Привет мир" in content
    assert "<v SPEAKER_01>Как дела" in content
    assert "00:00:00.000 --> 00:00:05.000" in content


def test_json_format(tmp_path: Path):
    """JSON writer produces valid structured JSON."""
    result = _make_result()
    out = tmp_path / "test.json"

    write_json(result, out)

    data = json.loads(out.read_text())
    assert data["audio"] == "/path/to/audio.wav"
    assert data["duration_sec"] == 10.5
    assert data["language"] == "Russian"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["speaker"] == "SPEAKER_00"
    assert len(data["segments"][0]["words"]) == 2
    assert data["speakers"] == ["SPEAKER_00", "SPEAKER_01"]


def test_get_writer():
    """get_writer returns correct writer functions."""
    assert get_writer("txt") is write_txt
    assert get_writer("srt") is write_srt
    assert get_writer("vtt") is write_vtt
    assert get_writer("json") is write_json


def test_get_writer_invalid():
    """get_writer raises ValueError for unknown format."""
    with pytest.raises(ValueError, match="Unknown format"):
        get_writer("unknown")
