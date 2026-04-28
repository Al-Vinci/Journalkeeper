import queue
import threading

from openai import OpenAI

from config import OPENAI_API_KEY_JOURNAL
from text_cleanup import cleanup_transcript_text


# Skapar en OpenAI-klient som används för live-transkribering.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)

# audio_queue innehåller ljudchunkar som ska transkriberas.
# text_queue innehåller färdig text som ska visas i gränssnittet.
audio_queue = queue.Queue()
text_queue = queue.Queue()


# worker kör i bakgrunden och tar emot ljudchunkar en i taget.
# Varje chunk transkriberas, städas och läggs sedan i textkön.
def worker():
    while True:
        audio_chunk = audio_queue.get()
        print("Worker received audio")
        if audio_chunk is None:
            break

        try:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_chunk,
                language="sv",
                prompt="Detta ar svensk veterinardiktering. Transkribera bara det som faktiskt sags.",
            )
            if transcript.text and transcript.text.strip():
                cleaned_text = cleanup_transcript_text(transcript.text)
                if cleaned_text:
                    text_queue.put(cleaned_text)
        except Exception as e:
            text_queue.put(f"[Fel]: {e}")


# Startar bakgrundstråden direkt när modulen laddas så att live-transkribering kan börja utan extra setup.
thread = threading.Thread(target=worker, daemon=True)
thread.start()
