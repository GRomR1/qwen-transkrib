"""Russian audio transcription with speaker diarization and punctuation restoration."""

__version__ = "0.1.0"

from qwen_transkrib.asr import transcribe_file
from qwen_transkrib.config import Settings
from qwen_transkrib.diarize import diarize_file
from qwen_transkrib.merge import assign_speakers
from qwen_transkrib.punctuate import PunctuationModel
from qwen_transkrib.schemas import Segment, TranscriptionResult, Word

__all__ = [
    "PunctuationModel",
    "Segment",
    "Settings",
    "TranscriptionResult",
    "Word",
    "assign_speakers",
    "diarize_file",
    "transcribe_file",
]
