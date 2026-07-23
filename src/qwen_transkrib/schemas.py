"""Pydantic models for transcription results."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Word(BaseModel):
    """A single word with timestamps."""

    text: str
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")


class Segment(BaseModel):
    """An utterance with speaker label and words."""

    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    speaker: str
    text: str
    words: list[Word] = Field(default_factory=list)


class TranscriptionResult(BaseModel):
    """Complete transcription result with metadata."""

    audio_path: str
    duration_sec: float = Field(ge=0)
    language: str
    model: str
    diarization_model: str | None = None
    speakers: list[str] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    full_text: str = ""

    def write_srt(self, path: str | Path) -> None:
        """Write SRT subtitle file."""
        from qwen_transkrib.writers.srt import write_srt

        write_srt(self, path)

    def write_vtt(self, path: str | Path) -> None:
        """Write WebVTT subtitle file."""
        from qwen_transkrib.writers.vtt import write_vtt

        write_vtt(self, path)

    def write_json(self, path: str | Path) -> None:
        """Write structured JSON file."""
        from qwen_transkrib.writers.json import write_json

        write_json(self, path)

    def write_txt(self, path: str | Path) -> None:
        """Write plain text file."""
        from qwen_transkrib.writers.txt import write_txt

        write_txt(self, path)
