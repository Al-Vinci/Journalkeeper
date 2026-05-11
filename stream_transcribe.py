import queue
import threading

from transcribe import transcribe_audio_result_fileobj


# audio_queue innehåller ljudchunkar som ska transkriberas.
# text_queue innehåller transkriberad text som ska visas i gränssnittet.
audio_queue = queue.Queue()
text_queue = queue.Queue()
use_diarization = False
settings_lock = threading.Lock()


def set_use_diarization(enabled):
    global use_diarization
    with settings_lock:
        use_diarization = enabled


def get_use_diarization():
    with settings_lock:
        return use_diarization


# worker kör i bakgrunden och tar emot ljudchunkar en i taget.
# Varje chunk transkriberas och diariseras bara om det är påslaget i appen.
def worker():
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break

        try:
            result = transcribe_audio_result_fileobj(
                audio_chunk,
                use_diarization=get_use_diarization(),
            )
            if result["journal_text"]:
                text_queue.put(result)
        except Exception as e:
            text_queue.put({"error": f"[Fel]: {e}"})


# Startar bakgrundstråden direkt när modulen laddas så att live-transkribering kan börja utan extra setup.
thread = threading.Thread(target=worker, daemon=True)
thread.start()
