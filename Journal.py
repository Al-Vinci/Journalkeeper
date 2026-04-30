import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer

from audio import cleanup_file, clear_saved_wavs, list_saved_wavs, save_temp_audio
from audio_stream import AudioProcessor
from generate import generate_journal
from stream_transcribe import text_queue
from transcribe import transcribe_audio


# Sätter grundinställningar för Streamlit-sidan.
st.set_page_config(page_title="Vet Journal AI", layout="wide")

# Visar rubrik och en tydlig varning om att texten bara är ett utkast.
st.title("Veterinarjournal - AI-utkast")
st.warning("Detta är ett UTKAST. Måste granskas av veterinär innan användning.")

# Den här delen av sidan hanterar live-transkribering från mikrofonen.
st.subheader("Live transkribering")

# Sparar användarens mikrofonval och live-text i sessionen så att de finns kvar mellan reruns.
if "echo_cancellation" not in st.session_state:
    st.session_state.echo_cancellation = False
if "noise_suppression" not in st.session_state:
    st.session_state.noise_suppression = False
if "auto_gain_control" not in st.session_state:
    st.session_state.auto_gain_control = False
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "live_plain_text" not in st.session_state:
    st.session_state.live_plain_text = ""
if "live_role_map" not in st.session_state:
    st.session_state.live_role_map = {}
if "live_errors" not in st.session_state:
    st.session_state.live_errors = []


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

# Skapar tomma platser i gränssnittet där live-texten kan uppdateras.
live_diarized_output = st.empty()
live_plain_output = st.empty()

# Om inspelningen pågår hämtas nya textbitar från kön och läggs till i hela transkriptet.
# Här används nu diarization-resultat i stället för bara ren text för att även visa talarroll.
if webrtc_ctx.state.playing:
    while not text_queue.empty():
        item = text_queue.get()

        if item.get("error"):
            st.session_state.live_errors.append(item["error"])
            continue

        diarized_text = item.get("diarized_text", "").strip()
        plain_text = item.get("plain_text", "").strip()
        speaker_roles = item.get("speaker_roles", {})

        if diarized_text:
            if st.session_state.full_text:
                st.session_state.full_text += "\n" + diarized_text
            else:
                st.session_state.full_text = diarized_text

        if plain_text:
            if st.session_state.live_plain_text:
                st.session_state.live_plain_text += " " + plain_text
            else:
                st.session_state.live_plain_text = plain_text

        if speaker_roles:
            st.session_state.live_role_map.update(speaker_roles)

    if st.session_state.full_text:
        live_diarized_output.markdown(st.session_state.full_text)
    if st.session_state.live_plain_text:
        live_plain_output.caption(st.session_state.live_plain_text)

    time.sleep(0.2)
    st.rerun()
else:
    # Visar tidigare live-text även när inspelningen stoppat så att användaren kan läsa färdigt resultatet.
    if st.session_state.full_text:
        live_diarized_output.markdown(st.session_state.full_text)
    if st.session_state.live_plain_text:
        live_plain_output.caption(st.session_state.live_plain_text)

# Visar eventuella transkriberingsfel separat så att de inte blandas ihop med journaltexten.
if st.session_state.live_errors:
    st.subheader("Live felmeddelanden")
    for error_text in st.session_state.live_errors[-5:]:
        st.error(error_text)

# Låt användaren slå av eller på webbläsarens inbyggda ljudbehandling.
# Detta behövs eftersom vissa filter kan göra dikterat tal avhugget eller inkomplett.
with st.expander("Mikrofoninställningar", expanded=True):
    st.checkbox("Echo cancellation", key="echo_cancellation")
    st.checkbox("Noise suppression", key="noise_suppression")
    st.checkbox("Auto gain control", key="auto_gain_control")
    st.caption("Om ljudet blir avhugget eller pumpande: låt alla tre vara avstängda.")

# Den här delen visar sparade backupfiler från live-inspelning.
st.subheader("Sparade backupfiler")

# Tar bort alla sparade wav-filer om användaren klickar på knappen.
if st.button("Rensa sparade WAV-filer"):
    deleted_files = clear_saved_wavs()
    st.success(f"Raderade {deleted_files} sparade WAV-filer.")
    st.rerun()

# Hämtar alla sparade backupfiler så att de kan visas i appen.
saved_wavs = list_saved_wavs()

# Visar varje sparad fil och lägger till en enkel spelare så att filen kan lyssnas på.
if saved_wavs:
    for wav_path in saved_wavs:
        st.write(wav_path.name)
        st.audio(str(wav_path), format="audio/wav")
else:
    st.info("Inga sparade WAV-filer än.")

# Den här delen hanterar vanlig filuppladdning som alternativ till live-inspelning.
audio_file = st.file_uploader("Ladda upp ljud (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

# Stoppar resten av flödet tills en fil faktiskt valts.
if audio_file is None:
    st.info("Ladda upp en ljudfil för att börja")
    st.stop()

# Hämtar filändelsen för att kunna validera formatet.
file_extension = audio_file.name.split(".")[-1].lower()

# Blockerar format som inte stöds av systemet.
if file_extension not in ["wav", "mp3", "m4a"]:
    st.error("Endast wav/mp3/m4a stöds")
    st.stop()

# Sparar den uppladdade filen tillfälligt på disk så att den kan skickas till transkribering.
tmp_path = save_temp_audio(audio_file.read(), file_extension)

try:
    # Gör transkriberingen av ljudfilen och hämtar både ren text och diariserad text.
    st.info("Transkriberar...")
    transcription_result = transcribe_audio(tmp_path)
    text = transcription_result["plain_text"]
    diarized_text = transcription_result["diarized_text"]
    speaker_roles = transcription_result["speaker_roles"]

    # Visar den transkriberade texten för användaren.
    st.subheader("Transkribering")
    st.write(text)

    # Visar diariserad text om flera talare behöver skiljas åt.
    st.subheader("Diariserad transkribering")
    st.text_area("Talare och roller", diarized_text, height=220)

    # Visar den semantiska tolkningen av rollerna om modellen lyckades hitta dem.
    if speaker_roles:
        st.subheader("Identifierade roller")
        for speaker, role in speaker_roles.items():
            st.write(f"Talare {speaker}: {role}")

    # Skickar diariserad text vidare för att skapa ett journalutkast.
    # Detta gör att journalutkastet får bättre kontext om vem som sagt vad.
    st.info("Skapar journalutkast...")
    journal = generate_journal(diarized_text)

    # Visar utkastet i en redigerbar ruta så att användaren kan justera texten.
    st.subheader("Journalutkast (SOAP)")
    st.text_area("Redigera vid behov", journal, height=300)

    st.success("Klart! Kopiera texten manuellt till journalsystemet.")
finally:
    # Tar alltid bort den tillfälliga filen så att projektmappen inte fylls med temporära filer.
    cleanup_file(tmp_path)
