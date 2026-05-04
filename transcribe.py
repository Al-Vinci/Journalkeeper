from openai import OpenAI

from config import OPENAI_API_KEY_JOURNAL
from text_cleanup import cleanup_transcript_text


# Skapar en OpenAI-klient för ren transkribering.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)


# kör vanlig transkribering på ett filobjekt.
def transcribe_audio_fileobj(file_obj):
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=file_obj,
        language="sv",
        temperature=0,
    )
    return cleanup_transcript_text(transcript.text or "")


# kör vanlig transkribering på en ljudfil på disk.
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        return transcribe_audio_fileobj(f)


# Bygger ett gemensamt resultatformat för appen, med eller utan diarization.
def build_transcription_result(plain_text, diarization_result=None):
    diarization_result = diarization_result or {}
    diarized_text = diarization_result.get("diarized_text", "").strip()

    return {
        "text": plain_text,
        "journal_text": diarized_text or plain_text,
        "diarized_text": diarized_text,
        "speaker_roles": diarization_result.get("speaker_roles", {}),
        "segments": diarization_result.get("segments", []),
        "use_diarization": bool(diarized_text),
    }


# kör transkribering och valfri diarization på ett filobjekt.
def transcribe_audio_result_fileobj(file_obj, use_diarization=False):
    plain_text = transcribe_audio_fileobj(file_obj)

    if not use_diarization:
        return build_transcription_result(plain_text)

    from diarization import diarize_audio_fileobj

    file_obj.seek(0)
    diarization_result = diarize_audio_fileobj(file_obj)
    return build_transcription_result(plain_text, diarization_result)


# kör transkribering och valfri diarization på en ljudfil på disk.
def transcribe_audio_result(file_path, use_diarization=False):
    with open(file_path, "rb") as f:
        return transcribe_audio_result_fileobj(f, use_diarization=use_diarization)
