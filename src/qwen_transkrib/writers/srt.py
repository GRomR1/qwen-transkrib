"""SRT subtitle writer."""

from __future__ import annotations

from pathlib import Path

from qwen_transkrib.schemas import TranscriptionResult


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(result: TranscriptionResult, path: str | Path) -> None:
    """Write transcription as SRT subtitle file.

    Format:
        1
        00:00:00,354 --> 00:00:12,990
        [SPEAKER_00] Text of first utterance

        2
        00:00:13,100 --> 00:00:25,500
        [SPEAKER_01] Text of second utterance
    """
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.segments, 1):
            start = _format_srt_time(seg.start)
            end = _format_srt_time(seg.end)
            f.write(f"{i}\n{start} --> {end}\n[{seg.speaker}] {seg.text}\n\n")
