"""Output format writers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qwen_transkrib.schemas import TranscriptionResult
from qwen_transkrib.writers.json import write_json
from qwen_transkrib.writers.srt import write_srt
from qwen_transkrib.writers.txt import write_txt
from qwen_transkrib.writers.vtt import write_vtt

_WRITERS: dict[str, Callable[[TranscriptionResult, Path], None]] = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
}


def get_writer(fmt: str) -> Callable[[TranscriptionResult, Path], None]:
    """Get writer function by format name.

    Args:
        fmt: Format name (txt, srt, vtt, json).

    Returns:
        Writer function.

    Raises:
        ValueError: If format is not supported.
    """
    if fmt not in _WRITERS:
        raise ValueError(f"Unknown format: {fmt}. Supported: {list(_WRITERS)}")
    return _WRITERS[fmt]


__all__ = ["get_writer", "write_json", "write_srt", "write_txt", "write_vtt"]
