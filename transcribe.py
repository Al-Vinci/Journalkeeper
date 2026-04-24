from openai import OpenAI
from config import OPENAI_API_KEY_JOURNAL

client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)

def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            language="sv",
            prompt="Detta är svensk veterinärdiktering. Transkribera bara det som faktiskt sägs."
        )
    return transcript.text
