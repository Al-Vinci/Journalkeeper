from openai import OpenAI

from config import OPENAI_API_KEY_JOURNAL
from text_cleanup import cleanup_transcript_text


# Skapar en OpenAI-klient för vanlig filbaserad transkribering.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)


# Tar en ljudfil från disk, transkriberar den på svenska och gör en lätt efterstäding av texten.
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            language="sv",
            prompt="Detta ar svensk veterinardiktering. Transkribera bara det som faktiskt sags.",
        )
    return cleanup_transcript_text(transcript.text or "")
