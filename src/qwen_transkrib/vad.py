"""Voice Activity Detection for audio chunking.

Uses Silero VAD to find speech segments and silence gaps,
then splits audio into chunks at natural pause boundaries.

Benefits over fixed-length chunking:
- Words aren't split mid-phrase
- Handles any audio length
- Natural pause boundaries improve ASR accuracy
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target_sr if needed."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def detect_speech_segments(
    audio: np.ndarray,
    sr: int,
    min_silence_ms: int = 500,
    min_speech_ms: int = 250,
    max_segment_sec: float = 60.0,
) -> list[tuple[float, float]]:
    """Detect speech segments using Silero VAD.

    Returns list of (start_sec, end_sec) tuples grouped into windows
    of max_segment_sec length, split at silence boundaries.
    """
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    audio_t = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(
        audio_t,
        vad,
        sampling_rate=sr,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
    )

    # Convert sample indices to seconds
    speech = [(t["start"] / sr, t["end"] / sr) for t in timestamps]

    # Group close segments into windows ≤ max_segment_sec
    windows: list[tuple[float, float]] = []
    if not speech:
        return windows

    cur_start, cur_end = speech[0]
    for s, e in speech[1:]:
        gap = s - cur_end
        # Merge if gap < 1s AND combined duration ≤ max_segment_sec
        if (cur_end - cur_start) + (e - s) + gap <= max_segment_sec and gap < 1.0:
            cur_end = e
        else:
            windows.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    windows.append((cur_start, cur_end))

    # Split any windows that exceed max_segment_sec
    split_windows: list[tuple[float, float]] = []
    for start, end in windows:
        while end - start > max_segment_sec:
            # Find a split point near the middle
            mid = start + max_segment_sec / 2
            split_windows.append((start, mid))
            start = mid
        split_windows.append((start, end))

    return split_windows


def extract_audio_chunks(
    audio_path: str | Path,
    windows: list[tuple[float, float]],
    output_dir: str | Path,
) -> list[tuple[float, float, Path]]:
    """Extract audio chunks for each VAD window.

    Returns list of (start_sec, end_sec, chunk_path) tuples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find ffmpeg
    ffmpeg = "ffmpeg"
    for candidate in [
        "/home/agent/.local/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2",
        "/opt/maca/ffmpeg/bin/ffmpeg",
    ]:
        if Path(candidate).exists():
            ffmpeg = candidate
            break

    chunks: list[tuple[float, float, Path]] = []
    for i, (start, end) in enumerate(windows):
        chunk_path = output_dir / f"chunk_{i:03d}.wav"
        duration = end - start
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(chunk_path),
        ]
        subprocess.run(cmd, check=True)
        chunks.append((start, end, chunk_path))

    return chunks
