"""Plain text writer."""

from __future__ import annotations

from pathlib import Path

from qwen_transkrib.schemas import TranscriptionResult


def write_txt(result: TranscriptionResult, path: str | Path) -> None:
    """Write transcription as plain text with speaker labels.

    Format:
        [SPEAKER_00] text of first utterance
        [SPEAKER_01] text of second utterance
    """
    with open(path, "w", encoding="utf-8") as f:
        for seg in result.segments:
            f.write(f"[{seg.speaker}] {seg.text}\n")
