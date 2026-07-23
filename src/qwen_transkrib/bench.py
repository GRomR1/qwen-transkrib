"""ASR benchmark against reference datasets.

Reads HF parquet files directly with pyarrow — bypasses Dataset Audio
feature (requires torchcodec, broken on MetaX / MACA).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files


def _strip_punct(text: str) -> str:
    """Remove punctuation and collapse whitespace for fair WER comparison."""
    import re
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def run_bench(
    dataset_name: str,
    split: str,
    transcribe_fn: Callable[[str], str],
    max_samples: int | None = None,
) -> dict:
    """Download HF dataset parquet, transcribe samples, compute WER.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split ('test', 'validation', etc.).
        transcribe_fn: Function(WAV_path) -> recognized text.
        max_samples: Limit samples (None = all).

    Returns:
        Dict with wer, sub, ins, del, ref_words, samples.
    """
    try:
        from jiwer import process_words
    except ImportError as e:
        raise ImportError(f"Missing package: {e}. Install: pip install jiwer")

    # Find parquet files for this split
    all_files = list_repo_files(dataset_name, repo_type="dataset")
    split_files = sorted(f for f in all_files if f.startswith(f"data/{split}") and f.endswith(".parquet"))
    if not split_files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' in {dataset_name}")

    total = {"sub": 0, "ins": 0, "del": 0, "ref": 0}
    processed = 0

    for parquet_path in split_files:
        if max_samples and processed >= max_samples:
            break

        # Download just this parquet file
        local = hf_hub_download(
            repo_id=dataset_name,
            filename=parquet_path,
            repo_type="dataset",
        )
        table = pq.read_table(local, columns=None)  # read all columns

        # Detect text column
        col_names = table.column_names
        text_col = next((c for c in col_names if c in ("transcription", "text", "sentence")), None)
        if text_col is None:
            continue

        for i in range(table.num_rows):
            if max_samples and processed >= max_samples:
                break

            ref = (table.column(text_col)[i].as_py() or "").strip()

            # Extract audio bytes from struct<bytes: binary, path: string>
            audio_struct = table.column("audio")[i].as_py()
            if not audio_struct:
                continue
            audio_bytes = audio_struct.get("bytes") or audio_struct.get("path")
            if not audio_bytes:
                continue
            if isinstance(audio_bytes, str):
                # It's a path — read the file
                with open(audio_bytes, "rb") as f:
                    audio_bytes = f.read()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
                f.write(audio_bytes)

            try:
                hyp = transcribe_fn(tmp)
                # Strip punctuation for fair WER (models may differ in punct output)
                ref_clean = _strip_punct(ref)
                hyp_clean = _strip_punct(hyp)
                m = process_words(ref_clean, hyp_clean)
                total["sub"] += m.substitutions
                total["ins"] += m.insertions
                total["del"] += m.deletions
                total["ref"] += sum(len(r) for r in m.references)
            finally:
                Path(tmp).unlink(missing_ok=True)

            processed += 1

        # Clean up downloaded parquet
        Path(local).unlink(missing_ok=True)

    n_err = total["sub"] + total["ins"] + total["del"]
    d = total["ref"]
    return {
        "wer": round(n_err / d * 100, 2) if d else 0.0,
        "substitutions": total["sub"],
        "insertions": total["ins"],
        "deletions": total["del"],
        "ref_words": d,
        "samples": processed,
    }
