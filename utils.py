import soundfile as sf
from io import BytesIO

def calculate_audio_duration_from_bytes(flac_data_bytes):
    with sf.SoundFile(BytesIO(flac_data_bytes)) as f:
        return len(f) / f.samplerate