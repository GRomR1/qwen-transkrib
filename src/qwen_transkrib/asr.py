"""ASR module: Qwen3-ASR and GigaAM backends."""

from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

from qwen_transkrib.config import Settings
from qwen_transkrib.schemas import Word

logger = logging.getLogger(__name__)

_model_cache: dict[str, Qwen3ASRModel] = {}
_gigaam_cache: dict[str, object] = {}

_attn_cache: dict[str, str] = {}


class ASRBackend(ABC):
    """Abstract ASR backend — implement per model family."""

    @abstractmethod
    def transcribe(self, path: str) -> str:
        """Transcribe audio file and return recognized text."""

    @abstractmethod
    def transcribe_words(self, path: str, context: str = "") -> tuple[list[Word], str]:
        """Transcribe with word-level timestamps.

        Args:
            path: Path to audio file.
            context: Optional hotwords for term correction.

        Returns:
            Tuple of (words, full_text).
        """


class Qwen3Backend(ASRBackend):
    """Qwen3-ASR backend (default)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = _get_qwen_model(settings)

    def transcribe(self, path: str) -> str:
        kwargs: dict = {
            "audio": path,
            "language": self.settings.language,
            "return_time_stamps": True,
        }
        results = self.model.transcribe(**kwargs)
        return results[0].text

    def transcribe_words(self, path: str, context: str = "") -> tuple[list[Word], str]:
        """Transcribe with word-level timestamps."""
        model = _get_qwen_model(self.settings)
        kwargs: dict = {
            "audio": path,
            "language": self.settings.language,
            "return_time_stamps": True,
        }
        if context:
            kwargs["context"] = context
        results = model.transcribe(**kwargs)
        r = results[0]
        words = [
            Word(text=item.text, start=item.start_time, end=item.end_time)
            for item in r.time_stamps.items
            if item.end_time > item.start_time
        ]
        return words, r.text


class GigaAMBackend(ASRBackend):
    """GigaAM ASR backend.

    Uses e2e_rnnt by default (RNNT decoder is 5x better than CTC on complex
    Russian texts — 2.6% vs 13.2% WER). Built-in punctuation and capitalization.

    Long audio is chunked at natural pause boundaries with overlap,
    end-to-end, and duplicate overlap regions are removed to avoid boundary errors.
    """

    MAX_SHORT_SEC = 24  # model threshold is 25s, leave margin
    DEFAULT_OVERLAP_SEC = 1.0  # overlap between chunks to reduce boundary errors

    def __init__(
        self,
        revision: str = "e2e_rnnt",
        device: str = "cuda:0",
    ) -> None:
        self.revision = revision
        self.device = device
        self._hf_model = _get_gigaam_hf_model(revision)
        self._native: object | None = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, path: str) -> str:
        """Transcribe file to text (handles long audio)."""
        # Try native gigaam first (better quality)
        try:
            model = self._get_native_model()
            return self._native_transcribe(model, path)
        except (ImportError, Exception) as e:
            logger.info("Native gigaam not available (%s), using HF transformers", e)
            return self._hf_transcribe(path)

    def transcribe_words(self, path: str, context: str = "") -> tuple[list[Word], str]:
        """Transcribe with word-level timestamps.

        Uses VAD-aware chunking with context carry-over for long audio.
        Uses native gigaam API when available for precise timestamps.
        Falls back to HF transformers with approximate timestamps otherwise.

        Args:
            path: Path to audio file.
            context: Optional hotwords for term correction.

        Returns:
            Tuple of (words, full_text).
        """
        audio, sr = _load_audio(path)
        duration = len(audio) / sr

        # Short audio: single pass (best quality)
        max_samp = self.MAX_SHORT_SEC * sr
        if len(audio) <= max_samp:
            return self._transcribe_short(path, audio, sr, duration, context)

        # Long audio: VAD-aware chunking with context carry-over
        return self._transcribe_long_vad(path, audio, sr, duration, context)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vad_chunked_transcribe(self, path: str) -> str:
        """Transcribe via HF transformers with VAD-aware chunking."""
        audio, sr = _load_audio(path)

        from qwen_transkrib.vad import detect_speech_segments

        windows = detect_speech_segments(
            audio, sr,
            min_silence_ms=500,
            max_segment_sec=float(self.MAX_SHORT_SEC),
        )

        if not windows:
            # No speech — single pass
            return self._hf_model.transcribe(path)

        import soundfile as sf

        texts: list[str] = []
        prev_text = ""
        overlap_sec = self.DEFAULT_OVERLAP_SEC

        with tempfile.TemporaryDirectory(prefix="gigaam_vad_") as tmpdir:
            for i, (start_sec, end_sec) in enumerate(windows):
                # Add overlap at the start (from previous chunk)
                chunk_start = max(0.0, start_sec - overlap_sec if i > 0 else start_sec)
                chunk_end = end_sec

                start_samp = int(chunk_start * sr)
                end_samp = int(chunk_end * sr)
                chunk = audio[start_samp:end_samp]
                chunk_path = f"{tmpdir}/chunk_{i:03d}.wav"
                sf.write(chunk_path, chunk, sr)

                chunk_text = self._hf_model.transcribe(chunk_path)

                # Overlap dedup: if this chunk overlaps with previous, strip
                # words that duplicate the end of the previous chunk
                if i > 0 and prev_text:
                    chunk_text = _strip_overlap(prev_text, chunk_text)

                texts.append(chunk_text)
                prev_text = chunk_text[-200:]  # carry over last 200 chars

        return " ".join(texts)

    def _transcribe_short(
        self, path: str, audio: np.ndarray, sr: int, duration: float, context: str
    ) -> tuple[list[Word], str]:
        """Transcribe short audio (<=24s) in a single pass."""
        try:
            model = self._get_native_model()
            return self._native_words(model, path)
        except ImportError:
            logger.info("Native gigaam not available, using HF transformers")
            return self._hf_words(path, audio, sr, duration, context)

    def _transcribe_long_vad(
        self, path: str, audio: np.ndarray, sr: int, duration: float, context: str
    ) -> tuple[list[Word], str]:
        """Transcribe long audio using VAD-aware chunking with context carry-over.

        Key improvements over fixed-length chunking:
        - Chunks at natural pause boundaries (VAD) instead of fixed 24s
        - Overlap between chunks to reduce boundary errors
        - Context carry-over: end of previous chunk text guides next chunk
        - Duplicate removal at overlap boundaries
        """
        from qwen_transkrib.vad import detect_speech_segments

        # Get VAD segments
        windows = detect_speech_segments(
            audio, sr,
            min_silence_ms=500,
            max_segment_sec=float(self.MAX_SHORT_SEC),
        )

        if not windows:
            # No speech detected — fall back to fixed chunking
            return self._transcribe_long_fixed(path, audio, sr, duration, context)

        all_words: list[Word] = []
        texts: list[str] = []
        prev_text = context  # context carry-over
        overlap_sec = self.DEFAULT_OVERLAP_SEC

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="gigaam_vad_") as tmpdir:
            for i, (start_sec, end_sec) in enumerate(windows):
                # Add overlap at the start (from previous chunk)
                chunk_start = max(0.0, start_sec - overlap_sec if i > 0 else start_sec)
                chunk_end = end_sec
                chunk_dur = chunk_end - chunk_start

                # Extract chunk
                start_samp = int(chunk_start * sr)
                end_samp = int(chunk_end * sr)
                chunk_audio = audio[start_samp:end_samp]
                chunk_path = f"{tmpdir}/chunk_{i:03d}.wav"
                sf.write(chunk_path, chunk_audio, sr)

                # Transcribe
                try:
                    model = self._get_native_model()
                    words, text = self._native_words(model, chunk_path)
                except ImportError:
                    words, text = self._hf_words(chunk_path, chunk_audio, sr, chunk_dur, prev_text)

                # Filter overlap duplicates BEFORE adding to global list.
                # Only drop words from the beginning of this chunk that
                # duplicate the end of the previous chunk.
                if i > 0 and words:
                    cutoff = start_sec - overlap_sec * 0.5
                    words = [w for w in words if (w.start + chunk_start) >= cutoff or len(words) <= 2]

                # Offset timestamps to global time
                for w in words:
                    all_words.append(Word(
                        text=w.text,
                        start=w.start + chunk_start,
                        end=w.end + chunk_start,
                    ))

                texts.append(text)
                prev_text = text[-200:]  # carry over last 200 chars as context

        full_text = " ".join(texts)
        return all_words, full_text

    def _transcribe_long_fixed(
        self, path: str, audio: np.ndarray, sr: int, duration: float, context: str
    ) -> tuple[list[Word], str]:
        """Fallback: fixed-length chunking when VAD finds no segments."""
        max_samp = self.MAX_SHORT_SEC * sr
        all_words: list[Word] = []
        texts: list[str] = []
        overlap = int(0.5 * sr)
        step = max_samp - overlap
        start = 0
        prev_text = context

        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="gigaam_fixed_") as tmpdir:
            while start < len(audio):
                end = min(start + max_samp, len(audio))
                chunk = audio[start:end]
                chunk_path = f"{tmpdir}/chunk_{start}.wav"
                sf.write(chunk_path, chunk, sr)

                try:
                    model = self._get_native_model()
                    words, text = self._native_words(model, chunk_path)
                except ImportError:
                    chunk_dur = len(chunk) / sr
                    words, text = self._hf_words(chunk_path, chunk, sr, chunk_dur, prev_text)

                for w in words:
                    all_words.append(Word(
                        text=w.text,
                        start=w.start + start / sr,
                        end=w.end + start / sr,
                    ))
                texts.append(text)
                prev_text = text[-200:]

                if end >= len(audio):
                    break
                start += step

        return all_words, " ".join(texts)

    def _get_native_model(self) -> object:
        """Lazy-init native gigaam model (for word timestamps)."""
        if self._native is not None:
            return self._native
        try:
            import gigaam
            # Map revision names: hf "e2e_rnnt" → native "v3_e2e_rnnt"
            native_revision = f"v3_{self.revision}" if not self.revision.startswith("v3_") else self.revision
            self._native = gigaam.load_model(native_revision)
            logger.info("GigaAM native model loaded: revision=%s", native_revision)
        except ImportError:
            raise ImportError(
                "gigaam package not installed. Run: uv sync --extra gigaam"
            )
        return self._native

    @staticmethod
    def _native_words(model: object, path: str) -> tuple[list[Word], str]:
        """Transcribe using native gigaam API and extract word timestamps."""
        r = model.transcribe(path, word_timestamps=True)
        words = [
            Word(text=w.text, start=w.start, end=w.end)
            for w in r.words
        ]
        return words, r.text

    def _native_transcribe(self, model: object, path: str) -> str:
        """Transcribe using native gigaam with VAD chunking for long audio."""
        audio, sr = _load_audio(path)

        # Short audio: single pass (best quality)
        if len(audio) <= self.MAX_SHORT_SEC * sr:
            r = model.transcribe(path)
            return r.text if hasattr(r, 'text') else r

        # Long audio: VAD chunking
        from qwen_transkrib.vad import detect_speech_segments
        windows = detect_speech_segments(
            audio, sr,
            min_silence_ms=500,
            max_segment_sec=float(self.MAX_SHORT_SEC),
        )

        if not windows:
            r = model.transcribe(path)
            return r.text if hasattr(r, 'text') else r

        import soundfile as sf
        texts: list[str] = []

        with tempfile.TemporaryDirectory(prefix="gigaam_native_") as tmpdir:
            for i, (start_sec, end_sec) in enumerate(windows):
                start_samp = int(start_sec * sr)
                end_samp = int(end_sec * sr)
                chunk = audio[start_samp:end_samp]
                chunk_path = f"{tmpdir}/chunk_{i:03d}.wav"
                sf.write(chunk_path, chunk, sr)

                r = model.transcribe(chunk_path)
                texts.append(r.text if hasattr(r, 'text') else r)

        return " ".join(texts)

    def _hf_transcribe(self, path: str) -> str:
        """Transcribe using HF transformers with VAD chunking."""
        audio, sr = _load_audio(path)

        # Short audio: single pass
        if len(audio) <= self.MAX_SHORT_SEC * sr:
            return self._hf_model.transcribe(path)

        return self._vad_chunked_transcribe(path)

    def _hf_words(
        self, path: str, audio: np.ndarray, sr: int, duration: float, context: str = ""
    ) -> tuple[list[Word], str]:
        """Transcribe using HF transformers with approximate word timestamps.

        Args:
            path: Path to audio file.
            audio: Audio array.
            sr: Sample rate.
            duration: Audio duration in seconds.
            context: Optional hotwords for term correction.
        """
        max_samples = 25 * 16000
        if len(audio) <= max_samples:
            text = self._hf_model.transcribe(path)
        else:
            texts: list[str] = []
            start = 0
            with tempfile.TemporaryDirectory(prefix="gigaam_") as tmpdir:
                while start < len(audio):
                    end = min(start + max_samples, len(audio))
                    chunk = audio[start:end]
                    chunk_path = f"{tmpdir}/chunk_{start}.wav"
                    import soundfile as sf
                    sf.write(chunk_path, chunk, sr)
                    texts.append(self._hf_model.transcribe(chunk_path))
                    if end >= len(audio):
                        break
                    start += max_samples - int(0.5 * sr)
            text = " ".join(texts)

        # Approximate word timestamps
        words = text.split()
        n = len(words)
        result = []
        for i, w in enumerate(words):
            start_t = i * duration / n if n else 0
            end_t = (i + 1) * duration / n if n else duration
            result.append(Word(text=w, start=start_t, end=end_t))
        return result, text


def _strip_overlap(prev_text: str, current_text: str) -> str:
    """Remove words duplicated at the start of current_text from overlap region.

    Finds the longest suffix of prev_text that matches a prefix of current_text
    and strips that prefix from current_text.
    """
    if not prev_text or not current_text:
        return current_text

    prev_words = prev_text.split()
    curr_words = current_text.split()
    if not prev_words or not curr_words:
        return current_text

    # Check overlap from longest possible down to 1 word
    max_overlap = min(len(prev_words), len(curr_words))
    for n in range(max_overlap, 0, -1):
        if prev_words[-n:] == curr_words[:n]:
            return " ".join(curr_words[n:])

    return current_text


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 numpy array."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _patch_gigaam_load_audio(model: object) -> None:
    """Patch GigaAM's load_audio to use soundfile instead of ffmpeg subprocess."""
    import sys
    import soundfile as sf
    import torch

    # Find the GigaAM module in sys.modules
    for mod_name, mod in sys.modules.items():
        if "gigaam" in mod_name.lower() and hasattr(mod, "load_audio"):
            original = mod.load_audio

            def patched_load_audio(
                audio_path: str, sample_rate: int = 16000, _orig=original
            ) -> torch.Tensor:
                try:
                    # Try soundfile first (fast, no subprocess)
                    audio, sr = sf.read(audio_path, dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    if sr != sample_rate:
                        # Resample if needed
                        import numpy as np
                        duration = len(audio) / sr
                        target_len = int(duration * sample_rate)
                        audio = np.interp(
                            np.linspace(0, len(audio), target_len, endpoint=False),
                            np.arange(len(audio)),
                            audio,
                        ).astype(np.float32)
                    return torch.from_numpy(audio)
                except Exception:
                    # Fallback to original ffmpeg-based loading
                    return _orig(audio_path, sample_rate)

            mod.load_audio = patched_load_audio
            logger.info("Patched GigaAM load_audio: soundfile instead of ffmpeg")
            break


def _get_gigaam_hf_model(revision: str = "e2e_rnnt") -> object:
    """Get or cache GigaAM model from HuggingFace (HF transformers path)."""
    key = f"ai-sage/GigaAM-v3:{revision}"
    if key not in _gigaam_cache:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "ai-sage/GigaAM-v3",
            revision=revision,
            trust_remote_code=True,
        )
        # Patch load_audio AFTER model is loaded (module is in sys.modules now)
        _patch_gigaam_load_audio(model)
        _gigaam_cache[key] = model
        logger.info("GigaAM HF model loaded: revision=%s", revision)
    return _gigaam_cache[key]


def create_backend(backend: str, settings: Settings | None = None) -> ASRBackend:
    """Factory: return the right ASR backend."""
    if backend == "gigaam":
        # e2e_rnnt: RNNT decoder is 5x better than CTC on complex Russian texts
        return GigaAMBackend(revision="e2e_rnnt")
    return Qwen3Backend(settings or Settings())


def _pick_attn_implementation(device: str) -> str:
    """Pick the best attention implementation for the current device.

    Priority: flash_attention_2 > sdpa > eager.
    flash_attention_2 used only on CUDA-capable devices with flash_attn installed.
    sdpa is built into PyTorch >= 2.0 and works on any device.
    """
    if device in _attn_cache:
        return _attn_cache[device]

    # flash_attention_2 — needs flash_attn package + CUDA-capable device
    if "cuda" in device or device == "cuda":
        try:
            import flash_attn  # noqa: F401
            impl = "flash_attention_2"
        except ImportError:
            impl = "sdpa"
    else:
        impl = "sdpa"

    _attn_cache[device] = impl
    logger.info("Attention implementation: %s (device=%s)", impl, device)
    return impl


def _get_qwen_model(settings: Settings) -> Qwen3ASRModel:
    """Get or create cached Qwen3 ASR model."""
    key = f"{settings.asr_model}:{settings.device}"
    if key not in _model_cache:
        attn_impl = _pick_attn_implementation(settings.device)
        _model_cache[key] = Qwen3ASRModel.from_pretrained(
            settings.asr_model,
            dtype=torch.bfloat16,
            device_map=settings.device,
            attn_implementation=attn_impl,
            max_inference_batch_size=settings.batch_size,
            max_new_tokens=settings.max_new_tokens,
            forced_aligner=settings.aligner_model,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=settings.device,
            ),
        )
        logger.info(
            "ASR model loaded: %s (device=%s, attn=%s)",
            settings.asr_model, settings.device, attn_impl,
        )
    return _model_cache[key]


def transcribe_file(
    path: Path,
    settings: Settings,
    context: str = "",
) -> tuple[list[Word], str, str]:
    """Transcribe audio file and return words with timestamps.

    Args:
        path: Path to audio file.
        settings: Application settings.
        context: Optional hotwords for term correction (e.g. "Google, Microsoft").

    Returns:
        Tuple of (words, full_text, language).
    """
    backend = Qwen3Backend(settings)
    words, text = backend.transcribe_words(str(path), context)
    return words, text, settings.language


def clear_cache() -> None:
    """Clear cached models to free memory."""
    _model_cache.clear()
    _gigaam_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_file_vad(
    path: Path,
    settings: Settings,
    context: str = "",
    min_silence_ms: int = 500,
    max_segment_sec: float = 60.0,
) -> tuple[list[Word], str, str]:
    """Transcribe audio using VAD-based chunking.

    Splits audio at natural pause boundaries using Silero VAD,
    transcribes each chunk independently, then merges results.

    Args:
        path: Path to audio file.
        settings: Application settings.
        context: Optional hotwords for term correction.
        min_silence_ms: Minimum silence duration to split (ms).
        max_segment_sec: Maximum chunk duration (seconds).

    Returns:
        Tuple of (words, full_text, language).
    """
    from qwen_transkrib.vad import detect_speech_segments, extract_audio_chunks, load_audio

    # Load audio
    audio, sr = load_audio(path)
    duration = len(audio) / sr
    logger.info("Audio loaded: %.1fs (%d Hz)", duration, sr)

    # Detect speech segments
    windows = detect_speech_segments(
        audio, sr,
        min_silence_ms=min_silence_ms,
        max_segment_sec=max_segment_sec,
    )
    logger.info("VAD: %d chunks (max %.0fs each)", len(windows), max_segment_sec)

    if not windows:
        return [], "", settings.language

    # Extract chunks to temp files
    with tempfile.TemporaryDirectory(prefix="qwen_vad_") as tmpdir:
        chunks = extract_audio_chunks(path, windows, tmpdir)

        # Transcribe each chunk
        backend = Qwen3Backend(settings)
        all_words: list[Word] = []
        full_texts: list[str] = []
        detected_language = settings.language

        for i, (start_sec, end_sec, chunk_path) in enumerate(chunks):
            chunk_words, chunk_text, chunk_lang = backend.transcribe_words(
                str(chunk_path), context
            )

            if i == 0:
                detected_language = chunk_lang

            # Offset timestamps to global time
            for w in chunk_words:
                all_words.append(
                    Word(
                        text=w.text,
                        start=w.start + start_sec,
                        end=w.end + start_sec,
                    )
                )

            full_texts.append(chunk_text)

            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                logger.info("ASR: %d/%d chunks", i + 1, len(chunks))

    full_text = " ".join(full_texts)
    logger.info("VAD transcription complete: %d words", len(all_words))

    return all_words, full_text, detected_language
