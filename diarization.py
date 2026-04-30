import json

from openai import OpenAI

from config import OPENAI_API_KEY_JOURNAL
from text_cleanup import cleanup_transcript_text


# Skapar en OpenAI-klient för diarization och semantisk rolltolkning.
client = OpenAI(api_key=OPENAI_API_KEY_JOURNAL)


# Försöker tolka vilken roll varje talare har i dialogen.
# Tanken är att ljudmodellen hittar vem som talar och GPT hjälper till att förstå rollen.
def infer_speaker_roles(segments):
    if not segments:
        return {}

    segment_lines = []
    for segment in segments:
        speaker = segment.get("speaker", "Okänd")
        text = segment.get("text", "").strip()
        if not text:
            continue
        segment_lines.append(f"{speaker}: {text}")

    if not segment_lines:
        return {}

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du analyserar en svensk veterinärdialog. "
                        "Du får talare märkta med bokstäver som A, B eller C. "
                        "Avgör om varje talare bäst motsvarar Veterinär, Djurägare eller Okänd. "
                        "Svara endast med strikt JSON på formen "
                        '{"A":"Veterinär","B":"Djurägare"}. '
                        "Om du är osäker ska du använda Okänd."
                    ),
                },
                {"role": "user", "content": "\n".join(segment_lines)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
        return {}
    except Exception:
        return {}


# Formaterar diariserade segment så att varje rad får en tydlig talarroll.
# Cleanup görs per segment för att minska brus utan att slå ihop talare.
def format_diarized_segments(segments, speaker_roles=None):
    speaker_roles = speaker_roles or {}
    formatted_lines = []

    for segment in segments:
        speaker = segment.get("speaker", "Okänd")
        text = (segment.get("text") or "").strip()
        if not text:
            continue

        role = speaker_roles.get(speaker, f"Talare {speaker}")
        cleaned_text = cleanup_transcript_text(text)
        if not cleaned_text:
            continue
        formatted_lines.append(f"{role}: {cleaned_text}")

    return "\n".join(formatted_lines).strip()


# Konverterar diarization-svaret till ett enklare internt format som resten av appen kan använda.
def build_diarization_result(transcript):
    segments = []
    for segment in transcript.segments:
        segments.append(
            {
                "speaker": segment.speaker,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            }
        )

    speaker_roles = infer_speaker_roles(segments)
    diarized_text = format_diarized_segments(segments, speaker_roles)
    plain_text = cleanup_transcript_text(transcript.text or "")

    return {
        "plain_text": plain_text,
        "diarized_text": diarized_text or plain_text,
        "speaker_roles": speaker_roles,
        "segments": segments,
    }


# Kör röstbaserad diarization på ett filobjekt och kompletterar med GPT-baserad rolltolkning.
# Denna används för live-chunkar där ljudet redan finns i minnet.
def transcribe_audio_fileobj_with_diarization(file_obj):
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe-diarize",
        file=file_obj,
        response_format="diarized_json",
        language="sv",
        chunking_strategy={
            "type": "server_vad",
            "prefix_padding_ms": 300,
            "silence_duration_ms": 450,
            "threshold": 0.5,
        },
        temperature=0,
    )
    return build_diarization_result(transcript)


# Kör röstbaserad diarization på en ljudfil på disk.
# Denna används för uppladdade filer och sparade backupfiler.
def transcribe_audio_with_diarization(file_path):
    with open(file_path, "rb") as f:
        return transcribe_audio_fileobj_with_diarization(f)
