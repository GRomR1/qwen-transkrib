"""Custom exceptions for russian-asr-diarize."""


class RuAsrError(Exception):
    """Base exception for all qwen_transkrib errors."""


class UnsupportedLanguage(RuAsrError):
    """Raised when the requested language is not supported by the ASR model."""


class ModelNotFoundError(RuAsrError):
    """Raised when a required model cannot be loaded."""


class AudioTooLong(RuAsrError):
    """Raised when audio exceeds maximum supported duration."""


class DiarizationError(RuAsrError):
    """Raised when speaker diarization fails."""


class OutputError(RuAsrError):
    """Raised when writing output files fails."""
