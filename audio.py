import tempfile
import os
import wave
from datetime import datetime
from pathlib import Path

RECORDINGS_DIR = Path("saved_wavs")

def save_wav(audio_bytes, filename="output.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(audio_bytes)
    return filename

def save_temp_audio(audio_bytes, extension):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
        tmp.write(audio_bytes)
        return tmp.name

def cleanup_file(path):
    if os.path.exists(path):
        os.remove(path)

def ensure_recordings_dir():
    RECORDINGS_DIR.mkdir(exist_ok=True)
    return RECORDINGS_DIR

def create_live_recording_path():
    recordings_dir = ensure_recordings_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return recordings_dir / f"live_recording_{timestamp}.wav"

def list_saved_wavs():
    recordings_dir = ensure_recordings_dir()
    return sorted(recordings_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)

def clear_saved_wavs():
    deleted = 0
    for wav_path in list_saved_wavs():
        try:
            wav_path.unlink(missing_ok=True)
            deleted += 1
        except PermissionError:
            continue
    return deleted
