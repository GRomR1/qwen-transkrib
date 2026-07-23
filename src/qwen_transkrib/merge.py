"""Speaker assignment via interval-overlap algorithm.

Ported from whisperx.diarize.assign_word_speakers.
Pure Python/NumPy implementation — no external dependencies.
"""

from __future__ import annotations

import pandas as pd

from qwen_transkrib.schemas import Segment, Word


def _overlap_duration(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate overlap duration between two intervals."""
    return max(0.0, min(end1, end2) - max(start1, start2))


def _merge_consecutive_turns(turns: pd.DataFrame, max_gap: float = 1.5) -> pd.DataFrame:
    """Merge consecutive turns of the same speaker separated by short gaps.

    This produces longer, more natural segments that follow speech flow.
    """
    if turns.empty:
        return turns

    merged = []
    cur_start, cur_end, cur_speaker = (
        turns.iloc[0]["start"],
        turns.iloc[0]["end"],
        turns.iloc[0]["speaker"],
    )

    for _, row in turns.iloc[1:].iterrows():
        if row["speaker"] == cur_speaker and row["start"] - cur_end <= max_gap:
            cur_end = row["end"]
        else:
            merged.append({"start": cur_start, "end": cur_end, "speaker": cur_speaker})
            cur_start, cur_end, cur_speaker = row["start"], row["end"], row["speaker"]

    merged.append({"start": cur_start, "end": cur_end, "speaker": cur_speaker})
    return pd.DataFrame(merged)


def assign_speakers(
    words: list[Word],
    turns: pd.DataFrame,
    gap_threshold: float = 3.0,
) -> list[Segment]:
    """Assign speakers to words via max-overlap with diarization turns.

    Algorithm:
    1. Merge consecutive same-speaker turns (for longer natural segments)
    2. For each word, find the diarization turn with maximum overlap
    3. Assign that turn's speaker to the word
    4. Group consecutive same-speaker words into Segments

    Args:
        words: List of words with timestamps.
        turns: DataFrame with columns: start, end, speaker.
        gap_threshold: Maximum gap (seconds) between words in same segment.

    Returns:
        List of Segments with speaker labels.
    """
    if not words or turns.empty:
        return []

    # Merge consecutive same-speaker turns for longer segments
    merged_turns = _merge_consecutive_turns(turns)

    # Convert turns to list of tuples for faster iteration
    turn_list = [(row["start"], row["end"], row["speaker"]) for _, row in merged_turns.iterrows()]

    # Step 1: Assign speaker to each word
    assigned: list[tuple[Word, str]] = []
    for w in words:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        min_distance = float("inf")
        nearest_speaker = "UNKNOWN"

        for t_start, t_end, t_speaker in turn_list:
            overlap = _overlap_duration(w.start, w.end, t_start, t_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t_speaker

            # Track nearest turn for fallback
            if w.end < t_start:
                dist = t_start - w.end
            elif w.start > t_end:
                dist = w.start - t_end
            else:
                dist = 0.0
            if dist < min_distance:
                min_distance = dist
                nearest_speaker = t_speaker

        # Use overlap winner, or fallback to nearest if no overlap
        if best_overlap > 0:
            assigned.append((w, best_speaker))
        else:
            assigned.append((w, nearest_speaker))

    # Step 2: Group consecutive same-speaker words into Segments
    segments: list[Segment] = []
    current_words: list[Word] = []
    current_speaker: str | None = None

    for word, speaker in assigned:
        if speaker == current_speaker and current_words:
            # Check gap threshold
            gap = word.start - current_words[-1].end
            if gap <= gap_threshold:
                current_words.append(word)
                continue

        # Flush previous segment
        if current_words:
            segments.append(
                Segment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    speaker=current_speaker or "UNKNOWN",
                    text=" ".join(w.text for w in current_words),
                    words=current_words,
                )
            )

        current_words = [word]
        current_speaker = speaker

    # Flush last segment
    if current_words:
        segments.append(
            Segment(
                start=current_words[0].start,
                end=current_words[-1].end,
                speaker=current_speaker or "UNKNOWN",
                text=" ".join(w.text for w in current_words),
                words=current_words,
            )
        )

    return segments
