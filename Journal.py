import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer

from audio import cleanup_file, clear_saved_wavs, list_saved_wavs, save_temp_audio
from audio_stream import AudioProcessor
from generate import generate_journal
from stream_transcribe import set_use_diarization, text_queue
from transcribe import transcribe_audio_result


# Sätter grundinställningar för Streamlit-sidan.
st.set_page_config(page_title="Vet Journal AI", layout="wide")

# Visar rubrik och en tydlig varning om att texten bara är ett utkast.
st.title("Veterinarjournal - AI-utkast")
st.warning("Detta är ett UTKAST. Maste granskas av veterinar innan anvandning.")

# Den här delen av sidan hanterar live-transkribering från mikrofonen.
st.subheader("Live transkribering")

# Sparar användarens mikrofonval, live-text och journalutkast mellan reruns.
if "echo_cancellation" not in st.session_state:
    st.session_state.echo_cancellation = False
if "noise_suppression" not in st.session_state:
    st.session_state.noise_suppression = False
if "auto_gain_control" not in st.session_state:
    st.session_state.auto_gain_control = False
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "live_journal" not in st.session_state:
    st.session_state.live_journal = ""
if "live_errors" not in st.session_state:
    st.session_state.live_errors = []
if "use_diarization" not in st.session_state:
    st.session_state.use_diarization = False


st.checkbox("Anvand diarization", key="use_diarization")
set_use_diarization(st.session_state.use_diarization)

if st.button("Rensa live-transkribering"):
    st.session_state.full_text = ""
    st.session_state.live_journal = ""
    st.session_state.live_errors = []
    st.rerun()


# Startar WebRTC-förbindelsen som tar emot mikrofonljud i realtid.
# AudioProcessor gör själva bearbetningen av ljudet innan transkribering.
webrtc_ctx = webrtc_streamer(
    key="live3",
    audio_processor_factory=AudioProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "audio": {
            "echoCancellation": st.session_state.echo_cancellation,
            "noiseSuppression": st.session_state.noise_suppression,
            "autoGainControl": st.session_state.auto_gain_control,
        },
        "video": False,
    },
    async_processing=True,
)

# Skapar en tom plats i gränssnittet där live-texten kan uppdateras.
live_output = st.empty()

# Om inspelningen pågår hämtas nya textbitar från kön och läggs till i hela transkriptet.
if webrtc_ctx.state.playing:
    while not text_queue.empty():
        item = text_queue.get()

        if item.get("error"):
            st.session_state.live_errors.append(item["error"])
            continue

        transcript_text = item.get("journal_text", item.get("text", "")).strip()
        if transcript_text:
            if st.session_state.full_text:
                separator = "\n" if item.get("diarized_text") else " "
                st.session_state.full_text += separator + transcript_text
            else:
                st.session_state.full_text = transcript_text

    if st.session_state.full_text:
        live_output.markdown(st.session_state.full_text)

    time.sleep(0.2)
    st.rerun()
else:
    # Visar tidigare live-text även när inspelningen stoppat så att användaren kan läsa färdigt resultatet.
    if st.session_state.full_text:
        live_output.markdown(st.session_state.full_text)

# Visar eventuella transkriberingsfel separat så att de inte blandas ihop med journaltexten.
if st.session_state.live_errors:
    st.subheader("Live felmeddelanden")
    for error_text in st.session_state.live_errors[-5:]:
        st.error(error_text)

# gör om live-transkriberingen till journal när användaren är klar med inspelningen.
if st.session_state.full_text:
    if st.button("Skapa journal från live-transkribering"):
        st.info("Skapar journalutkast...")
        st.session_state.live_journal = generate_journal(st.session_state.full_text)

    if st.session_state.live_journal:
        st.subheader("Journalutkast från live-transkribering")
        st.text_area("Redigera vid behov", st.session_state.live_journal, height=300, key="live_journal_editor")

# Låt användaren slå av eller på webbläsarens inbyggda ljudbehandling.
# Detta behövs eftersom vissa filter kan göra dikterat tal avhugget eller inkomplett.
with st.expander("Mikrofoninställningar", expanded=True):
    st.checkbox("Echo cancellation", key="echo_cancellation")
    st.checkbox("Noise suppression", key="noise_suppression")
    st.checkbox("Auto gain control", key="auto_gain_control")
    st.caption("Om ljudet blir avhugget eller pumpande: Låt alla tre vara avstangda.")

# Den här delen visar sparade backupfiler från live-inspelning.
st.subheader("Sparade backupfiler")

# Tar bort alla sparade wav-filer om användaren klickar på knappen.
if st.button("Rensa sparade WAV-filer"):
    deleted_files = clear_saved_wavs()
    st.success(f"Raderade {deleted_files} sparade WAV-filer.")
    st.rerun()

# Hämtar alla sparade backupfiler så att de kan visas i appen.
saved_wavs = list_saved_wavs()

# Visar varje sparad fil och lägger till en enkel spelare så att filen kan lyssnas pa.
if saved_wavs:
    for wav_path in saved_wavs:
        st.write(wav_path.name)
        st.audio(str(wav_path), format="audio/wav")
else:
    st.info("Inga sparade WAV-filer an.")

# Den här delen hanterar vanlig filuppladdning som alternativ till live-inspelning.
audio_file = st.file_uploader("Ladda upp ljud (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

# Stoppar resten av filflodet tills en fil faktiskt valts.
if audio_file is None:
    st.info("Ladda upp en ljudfil för att transkribera och skapa journal.")
    st.stop()

# Hämtar filändelsen för att kunna validera formatet.
file_extension = audio_file.name.split(".")[-1].lower()

# Blockerar format som inte stods av systemet.
if file_extension not in ["wav", "mp3", "m4a"]:
    st.error("Endast wav/mp3/m4a stods")
    st.stop()

# Sparar den uppladdade filen tillfälligt på disk så att den kan skickas till transkribering.
tmp_path = save_temp_audio(audio_file.read(), file_extension)

try:
    # gör transkriberingen av ljudfilen.
    st.info("Transkriberar...")
    transcription_result = transcribe_audio_result(
        tmp_path,
        use_diarization=st.session_state.use_diarization,
    )
    text = transcription_result["text"]
    journal_text = transcription_result["journal_text"]
    diarized_text = transcription_result["diarized_text"]
    speaker_roles = transcription_result["speaker_roles"]

    # Visar den transkriberade texten för användaren.
    st.subheader("Transkribering")
    st.write(text)

    if diarized_text:
        st.subheader("Diariserad transkribering")
        st.text_area("Talare och roller", diarized_text, height=220)

    if speaker_roles:
        st.subheader("Identifierade roller")
        for speaker, role in speaker_roles.items():
            st.write(f"Talare {speaker}: {role}")

    # Skickar transkriberingen vidare för att skapa ett journalutkast.
    st.info("Skapar journalutkast...")
    journal = generate_journal(journal_text)

    # Visar utkastet i en redigerbar ruta så att användaren kan justera texten.
    st.subheader("Journalutkast")
    st.text_area("Redigera vid behov", journal, height=300, key="uploaded_journal_editor")

    st.success("Klart! Kopiera texten manuellt till journalsystemet.")
finally:
    # Tar alltid bort den tillfälliga filen så att projektmappen inte fylls med temporära filer.
    cleanup_file(tmp_path)
