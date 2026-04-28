import tempfile
import os
import wave
from datetime import datetime
from pathlib import Path


# Den här mappen används för att spara backupfiler från live-inspelning.
RECORDINGS_DIR = Path("saved_wavs")


# Sparar raka ljudbytes som en enkel wav-fil.
# Den här funktionen finns som hjälp om man vill skapa en fil direkt från PCM-data.
def save_wav(audio_bytes, filename="output.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(audio_bytes)
    return filename


# Sparar en uppladdad ljudfil till en temporär fil så att andra funktioner kan läsa den.
def save_temp_audio(audio_bytes, extension):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
        tmp.write(audio_bytes)
        return tmp.name


# Raderar en fil om den finns.
# Detta används framför allt för att städa bort temporära filer efter transkribering.
def cleanup_file(path):
    if os.path.exists(path):
        os.remove(path)


# Ser till att mappen för sparade backupfiler alltid finns innan den används.
def ensure_recordings_dir():
    RECORDINGS_DIR.mkdir(exist_ok=True)
    return RECORDINGS_DIR


# Skapar ett unikt filnamn för varje live-inspelning.
# Tidsstämpeln gör att filer inte skriver över varandra.
def create_live_recording_path():
    recordings_dir = ensure_recordings_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return recordings_dir / f"live_recording_{timestamp}.wav"


# Returnerar alla sparade wav-filer sorterade med senaste filen först.
def list_saved_wavs():
    recordings_dir = ensure_recordings_dir()
    return sorted(recordings_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)


# Rensar alla sparade wav-filer som inte är lästa av ett annat program.
def clear_saved_wavs():
    deleted = 0
    for wav_path in list_saved_wavs():
        try:
            wav_path.unlink(missing_ok=True)
            deleted += 1
        except PermissionError:
            continue
    return deleted
