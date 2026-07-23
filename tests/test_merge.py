"""Unit tests for speaker assignment (merge module)."""

from __future__ import annotations

import pandas as pd

from qwen_transkrib.merge import assign_speakers
from qwen_transkrib.schemas import Word


def test_single_speaker():
    """All words assigned to one speaker."""
    words = [Word(text="А", start=0.0, end=0.3), Word(text="Б", start=0.5, end=0.8)]
    turns = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": ["SPEAKER_00"]})

    segments = assign_speakers(words, turns)

    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[0].text == "А Б"


def test_two_speakers():
    """Words split across two speakers."""
    words = [
        Word(text="А", start=0.0, end=0.3),
        Word(text="Б", start=1.5, end=1.8),
    ]
    turns = pd.DataFrame(
        {
            "start": [0.0, 1.5],
            "end": [0.5, 2.0],
            "speaker": ["SPEAKER_00", "SPEAKER_01"],
        }
    )

    segments = assign_speakers(words, turns)

    assert len(segments) == 2
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[1].speaker == "SPEAKER_01"


def test_gap_threshold_split():
    """Words with large gap split into separate segments."""
    words = [
        Word(text="А", start=0.0, end=0.3),
        Word(text="Б", start=5.0, end=5.3),
    ]
    turns = pd.DataFrame(
        {
            "start": [0.0, 5.0],
            "end": [0.5, 5.5],
            "speaker": ["SPEAKER_00", "SPEAKER_00"],
        }
    )

    segments = assign_speakers(words, turns, gap_threshold=1.0)

    # Gap of 4.7s > threshold 1.0s, so two segments
    assert len(segments) == 2


def test_gap_within_threshold():
    """Words with small gap stay in same segment."""
    words = [
        Word(text="А", start=0.0, end=0.3),
        Word(text="Б", start=0.5, end=0.8),
    ]
    turns = pd.DataFrame(
        {
            "start": [0.0],
            "end": [1.0],
            "speaker": ["SPEAKER_00"],
        }
    )

    segments = assign_speakers(words, turns, gap_threshold=1.0)

    assert len(segments) == 1
    assert segments[0].text == "А Б"


def test_empty_words():
    """Empty word list returns empty segments."""
    turns = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": ["SPEAKER_00"]})
    segments = assign_speakers([], turns)
    assert segments == []


def test_empty_turns():
    """Empty turns returns empty segments."""
    words = [Word(text="А", start=0.0, end=0.3)]
    turns = pd.DataFrame(columns=["start", "end", "speaker"])
    segments = assign_speakers(words, turns)
    assert segments == []


def test_no_overlap():
    """Words with no overlapping turn get assigned to nearest speaker."""
    words = [Word(text="А", start=10.0, end=10.3)]
    turns = pd.DataFrame({"start": [0.0], "end": [1.0], "speaker": ["SPEAKER_00"]})

    segments = assign_speakers(words, turns)

    assert len(segments) == 1
    # Nearest speaker fallback: word at 10.0 is nearest to SPEAKER_00 (ends at 1.0)
    assert segments[0].speaker == "SPEAKER_00"


def test_multiple_words_per_speaker():
    """Multiple consecutive words from same speaker form one segment."""
    words = [
        Word(text="один", start=0.0, end=0.3),
        Word(text="два", start=0.4, end=0.7),
        Word(text="три", start=0.8, end=1.1),
    ]
    turns = pd.DataFrame({"start": [0.0], "end": [1.5], "speaker": ["SPEAKER_00"]})

    segments = assign_speakers(words, turns)

    assert len(segments) == 1
    assert segments[0].text == "один два три"
    assert len(segments[0].words) == 3


def test_overlap_duration():
    """Test _overlap_duration calculation."""
    from qwen_transkrib.merge import _overlap_duration

    # Partial overlap
    assert _overlap_duration(0, 5, 2, 7) == 3  # [0-5] ∩ [2-7] = 3s

    # Adjacent intervals (no overlap)
    assert _overlap_duration(0, 5, 5, 10) == 0

    # Disjoint intervals
    assert _overlap_duration(0, 5, 10, 15) == 0

    # Identical intervals
    assert _overlap_duration(0, 5, 0, 5) == 5

    # One contained in another
    assert _overlap_duration(0, 10, 2, 8) == 6

    # Partial overlap (reversed order)
    assert _overlap_duration(5, 10, 0, 7) == 2
