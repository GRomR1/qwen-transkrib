import sys as _sys
import torch
import soundfile as _sf
__version__ = "2.4.1-stub"
def load(uri, *a, **k):
    audio, sr = _sf.read(str(uri), dtype='float32', always_2d=False)
    if audio.ndim>1: audio=audio.mean(axis=1)
    return torch.from_numpy(audio).unsqueeze(0), sr
def save(uri, w, sr, *a, **k):
    if hasattr(w,'numpy'): w=w.numpy()
    _sf.write(str(uri), w, sr)
def info(uri):
    si = _sf.info(str(uri))
    class I: sample_rate=si.samplerate; num_channels=si.channels; num_frames=si.frames; duration=si.duration
    return I()
from . import transforms, functional, models, io, backend, compliance, sox_effects, utils
def set_audio_backend(n): pass
def get_audio_backend(): return 'soundfile'
def list_audio_backends(): return ['soundfile']
class _E:
    @staticmethod
    def init(): pass
