import os
import tempfile
import wave

import gradio as gr
import numpy as np

from generate import generate_journal
from transcribe import transcribe_audio_result


# Gradio kan lämna mikrofonljud som en numpy-array i stället för som en färdig ljudfil.
# OpenAI-transkriberingen arbetar stabilare när vi själva skriver en enkel wav-fil med känt format.
def write_temp_wav(sample_rate, audio_data):
    input_array = np.asarray(audio_data)

    # Gradio/browsern kan ge antingen normaliserade floatvärden (-1 till 1)
    # eller PCM-värden i ungefär int16-skala. Vi sparar detta innan senare steg,
    # eftersom stereo-mixning gör int16-data till float och annars kan feltolkas.
    input_is_float = np.issubdtype(input_array.dtype, np.floating)
    max_abs_input = np.max(np.abs(input_array)) if input_array.size else 0
    floats_are_normalized = input_is_float and max_abs_input <= 1.5

    audio_array = input_array.astype(np.float64)

    # Om ljudet är stereo mixas kanalerna till mono.
    # Det gör formatet enklare och matchar hur resten av projektets live-ljud hanteras.
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Sampla om till 16 kHz. Det minskar filstorleken och passar bra för taltranskribering.
    target_sample_rate = 16000
    if int(sample_rate) != target_sample_rate and audio_array.size > 0:
        duration = audio_array.size / float(sample_rate)
        target_size = max(1, int(duration * target_sample_rate))
        original_positions = np.linspace(0, audio_array.size - 1, num=audio_array.size)
        target_positions = np.linspace(0, audio_array.size - 1, num=target_size)
        audio_array = np.interp(target_positions, original_positions, audio_array)
        sample_rate = target_sample_rate

    # Skriv om ljudet till 16-bit PCM. Normaliserat floatljud multipliceras upp,
    # medan redan skalat PCM-ljud bara klipps till giltigt int16-intervall.
    if floats_are_normalized:
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
    else:
        audio_array = np.clip(audio_array, -32768, 32767).astype(np.int16)

    # Filen sparas temporärt eftersom transcribe.py förväntar sig en filväg eller ett filobjekt.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    # Vi skriver en enkel mono wav utan extra metadata för att undvika formatproblem mot API:t.
    with wave.open(temp_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(audio_array.tobytes())

    return temp_path


# Används i felmeddelanden så det går att se om ljudet är orimligt långt,
# har fel sample rate eller saknar förväntat wav-format.
def describe_audio_file(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return "ljudfil saknas"

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    try:
        with wave.open(audio_path, "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            duration = frames / float(sample_rate) if sample_rate else 0
        return f"{size_mb:.2f} MB, {duration:.1f} s, {sample_rate} Hz, {channels} kanal(er)"
    except Exception:
        return f"{size_mb:.2f} MB"


# Normaliserar Gradio-komponentens olika returformat till en vanlig filväg.
# För numpy-ljud skapas först en temporär wav-fil som tas bort efter transkribering.
def get_audio_path(audio_input):
    if isinstance(audio_input, str):
        return audio_input, False
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        sample_rate, audio_data = audio_input
        return write_temp_wav(sample_rate, audio_data), True
    if isinstance(audio_input, dict):
        path = audio_input.get("path") or audio_input.get("name")
        return path, False
    return None, False


# Tar bara bort filer som vi själva har skapat temporärt.
# Uppladdade filer som Gradio hanterar får Gradio själv städa upp.
def cleanup_temp_audio(audio_path, should_delete):
    if should_delete and audio_path and os.path.exists(audio_path):
        os.remove(audio_path)


# Gör rollmappningen från diarisation lättläst i gränssnittet.
def format_speaker_roles(speaker_roles):
    if not speaker_roles:
        return ""

    return "\n".join(
        f"Talare {speaker}: {role}"
        for speaker, role in speaker_roles.items()
    )


# Kör hela huvudflödet: ljud -> transkribering -> eventuell diarisation -> journalutkast.
# Transkribering och journalgenerering ligger i separata try-block så att en journalbugg
# inte raderar en lyckad transkribering från gränssnittet.
def transcribe_and_generate(audio_input, use_diarization):
    audio_path, should_delete_audio = get_audio_path(audio_input)
    if not audio_path:
        return "", "", "", "", "Ladda upp eller spela in ljud för att börja."

    try:
        transcription_result = transcribe_audio_result(
            audio_path,
            use_diarization=use_diarization,
        )

        text = transcription_result["text"]
        diarized_text = transcription_result["diarized_text"]
        speaker_roles = format_speaker_roles(transcription_result["speaker_roles"])
    except Exception as error:
        audio_info = describe_audio_file(audio_path)
        cleanup_temp_audio(audio_path, should_delete_audio)
        return "", "", "", "", f"Fel vid transkribering ({audio_info}): {type(error).__name__}: {error}"
    finally:
        cleanup_temp_audio(audio_path, should_delete_audio)

    journal_text = transcription_result["journal_text"]
    try:
        journal = generate_journal(journal_text) if journal_text else ""
        status = "Klart. Journalutkastet måste granskas av veterinär innan användning."
        return text, diarized_text, speaker_roles, journal, status
    except Exception as error:
        status = f"Transkribering klar, men fel vid journalgenerering: {type(error).__name__}: {error}"
        return text, diarized_text, speaker_roles, "", status


# Skapar en ny journal från text som användaren själv har korrigerat i rutorna.
# Om diarisation är på används den diariserade texten när den finns, annars vanlig transkribering.
def generate_from_edited_text(plain_text, diarized_text, use_diarization):
    source_text = diarized_text.strip() if use_diarization and diarized_text.strip() else plain_text.strip()
    if not source_text:
        return "", "Det finns ingen text att skapa journal från."

    try:
        journal = generate_journal(source_text)
        return journal, "Journalutkast skapat från redigerad text."
    except Exception as error:
        return "", f"Fel: {error}"


# Gradio-gränssnittet motsvarar ungefär Streamlit-appen:
# ljud spelas in/laddas upp först och bearbetas sedan när användaren klickar på knappen.
with gr.Blocks(title="Vet Journal AI") as demo:
    gr.Markdown("# Veterinärjournal - AI-utkast")
    gr.Markdown("Detta är ett utkast och måste granskas av veterinär innan användning.")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            format="wav",
            label="Spela in eller ladda upp ljud",
        )
        use_diarization = gr.Checkbox(
            label="Använd diarisation",
            value=False,
        )

    transcribe_button = gr.Button("Transkribera och skapa journal", variant="primary")
    status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        transcript_output = gr.Textbox(
            label="Transkribering",
            lines=12,
            interactive=True,
        )
        diarized_output = gr.Textbox(
            label="Diariserad transkribering",
            lines=12,
            interactive=True,
        )

    speaker_roles_output = gr.Textbox(
        label="Identifierade roller",
        lines=4,
        interactive=False,
    )

    journal_output = gr.Textbox(
        label="Journalutkast",
        lines=14,
        interactive=True,
    )

    generate_button = gr.Button("Skapa journal från redigerad text")

    transcribe_button.click(
        fn=transcribe_and_generate,
        inputs=[audio_input, use_diarization],
        outputs=[
            transcript_output,
            diarized_output,
            speaker_roles_output,
            journal_output,
            status_output,
        ],
    )

    generate_button.click(
        fn=generate_from_edited_text,
        inputs=[transcript_output, diarized_output, use_diarization],
        outputs=[journal_output, status_output],
    )


if __name__ == "__main__":
    demo.launch()
