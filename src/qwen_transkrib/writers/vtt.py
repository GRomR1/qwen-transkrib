"""WebVTT subtitle writer."""

from __future__ import annotations

from pathlib import Path

from qwen_transkrib.schemas import TranscriptionResult


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as VTT timestamp HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(result: TranscriptionResult, path: str | Path) -> None:
    """Write transcription as WebVTT file.

    Format:
        WEBVTT

        00:00:00.354 --> 00:00:12.990
        <v SPEAKER_00>Text of first utterance

        00:00:13.100 --> 00:00:25.500
        <v SPEAKER_01>Text of second utterance
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in result.segments:
            start = _format_vtt_time(seg.start)
            end = _format_vtt_time(seg.end)
            f.write(f"{start} --> {end}\n<v {seg.speaker}>{seg.text}\n\n")
