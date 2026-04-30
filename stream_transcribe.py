import queue
import threading

from diarization import transcribe_audio_fileobj_with_diarization


# audio_queue innehåller ljudchunkar som ska transkriberas.
# text_queue innehåller strukturerad text som ska visas i gränssnittet.
audio_queue = queue.Queue()
text_queue = queue.Queue()


# worker kör i bakgrunden och tar emot ljudchunkar en i taget.
# Varje chunk diariseras, rolltolkas och läggs sedan i textkön.
def worker():
    while True:
        audio_chunk = audio_queue.get()
        print("Worker received audio")
        if audio_chunk is None:
            break

        try:
            diarization_result = transcribe_audio_fileobj_with_diarization(audio_chunk)
            if diarization_result["diarized_text"]:
                text_queue.put(diarization_result)
        except Exception as e:
            text_queue.put({"error": f"[Fel]: {e}"})


# Startar bakgrundstråden direkt när modulen laddas så att live-transkribering kan börja utan extra setup.
thread = threading.Thread(target=worker, daemon=True)
thread.start()
