import queue
import threading
from openai import OpenAI
from config import OPENAI_API_KEY_JOURNAL

client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)

audio_queue = queue.Queue()
text_queue = queue.Queue()

def worker():
    while True:
        audio_chunk = audio_queue.get()
        print("🧠 Worker received audio")
        if audio_chunk is None:
            break

        try:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_chunk,
                language="sv",
                prompt="Detta är svensk veterinärdiktering. Transkribera bara det som faktiskt sägs."
            )
            if transcript.text and transcript.text.strip():
                text_queue.put(transcript.text.strip())
        except Exception as e:
            text_queue.put(f"[Fel]: {e}")

thread = threading.Thread(target=worker, daemon=True)
thread.start()
