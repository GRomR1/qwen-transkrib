"""Structured JSON writer."""

from __future__ import annotations

import json
from pathlib import Path

from qwen_transkrib.schemas import TranscriptionResult


def write_json(result: TranscriptionResult, path: str | Path) -> None:
    """Write transcription as structured JSON.

    Schema matches TASK_SPEC.md specification with audio metadata,
    speaker list, and segments containing words.
    """
    data = {
        "audio": result.audio_path,
        "duration_sec": result.duration_sec,
        "model": result.model,
        "language": result.language,
        "diarization_model": result.diarization_model,
        "speakers": result.speakers,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "words": [{"text": w.text, "start": w.start, "end": w.end} for w in seg.words],
            }
            for seg in result.segments
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
