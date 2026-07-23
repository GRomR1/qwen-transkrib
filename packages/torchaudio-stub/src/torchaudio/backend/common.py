class AudioMetaData:
    def __init__(self, sample_rate=16000, num_frames=0, num_channels=1, bits_per_sample=16, encoding="PCM_S"):
        self.sample_rate=sample_rate; self.num_frames=num_frames
        self.num_channels=num_channels; self.bits_per_sample=bits_per_sample
        self.encoding=encoding
def list_audio_backends(): return ["soundfile"]
def get_audio_backend(): return "soundfile"
def set_audio_backend(n): pass
