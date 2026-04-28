import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer

from audio import cleanup_file, clear_saved_wavs, list_saved_wavs, save_temp_audio
from audio_stream import AudioProcessor
from generate import generate_journal
from stream_transcribe import text_queue
from transcribe import transcribe_audio


# Sätter grundinställningar for Streamlit-sidan.
st.set_page_config(page_title="Vet Journal AI", layout="wide")

# Visar rubrik och en tydlig varning om att texten bara ar ett utkast.
st.title("Veterinarjournal - AI-utkast")
st.warning("Detta ar ett UTKAST. Maste granskas av veterinär innan anvandning.")

# Den här delen av sidan hanterar live-transkribering fran mikrofonen.
st.subheader("Live transkribering")

# Sparar användarens mikrofonval i sessionen så att de finns kvar mellan reruns.
if "echo_cancellation" not in st.session_state:
    st.session_state.echo_cancellation = False
if "noise_suppression" not in st.session_state:
    st.session_state.noise_suppression = False
if "auto_gain_control" not in st.session_state:
    st.session_state.auto_gain_control = False
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

# Låt användaren slå av eller på webbläsarens inbyggda ljudbehandling.
# Detta behövs eftersom vissa filter kan göra dikterat tal avhugget eller inkomplett.
with st.expander("Mikrofoninstallningar", expanded=True):
    st.checkbox("Echo cancellation", key="echo_cancellation")
    st.checkbox("Noise suppression", key="noise_suppression")
    st.checkbox("Auto gain control", key="auto_gain_control")
    st.caption("Om ljudet blir avhugget eller pumpande: lat alla tre vara avstangda.")

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
output = st.empty()

# Om inspelningen pågår hämtas nya textbitar från kön och laggs till i hela transkriptet.
# En kort sleep och rerun gör att sidan uppdateras kontinuerligt medan tal kommer in.
if webrtc_ctx.state.playing:
    new_texts = []

    while not text_queue.empty():
        new_texts.append(text_queue.get())

    if new_texts:
        st.session_state.full_text += " " + " ".join(new_texts)

    output.markdown(st.session_state.full_text)

    time.sleep(0.2)
    st.rerun()
elif st.session_state.full_text:
    # Visar tidigare transkriberad text när inspelningen har stoppat.
    output.markdown(st.session_state.full_text)

# Den här delen visar sparade backupfiler från live-inspelning.
st.subheader("Sparade backupfiler")

# Tar bort alla sparade wav-filer om användaren klickar pa knappen.
if st.button("Rensa sparade WAV-filer"):
    deleted_files = clear_saved_wavs()
    st.success(f"Raderade {deleted_files} sparade WAV-filer.")
    st.rerun()

# Hämta alla sparade backupfiler så att de kan visas i appen.
saved_wavs = list_saved_wavs()

# Visar varje sparad fil och lägger till en enkel spelare så att filen kan lyssnas på.
if saved_wavs:
    for wav_path in saved_wavs:
        st.write(wav_path.name)
        st.audio(str(wav_path), format="audio/wav")
else:
    st.info("Inga sparade WAV-filer an.")

# Den här delen hanterar vanlig filuppladdning som alternativ till live-inspelning.
audio_file = st.file_uploader("Ladda upp ljud (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

# Stoppar resten av flödet tills en fil faktiskt valts.
if audio_file is None:
    st.info("Ladda upp en ljudfil for att borja")
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
    # Gör transkriberingen av ljudfilen.
    st.info("Transkriberar...")
    text = transcribe_audio(tmp_path)

    # Visar den transkriberade texten för användaren.
    st.subheader("Transkribering")
    st.write(text)

    # Skickar transkriberingen vidare för att skapa ett journalutkast.
    st.info("Skapar journalutkast...")
    journal = generate_journal(text)

    # Visar utkastet i en redigerbar ruta så att användaren kan justera texten.
    st.subheader("Journalutkast (SOAP)")
    st.text_area("Redigera vid behov", journal, height=300)

    st.success("Klart! Kopiera texten manuellt till journalsystemet.")
finally:
    # Tar alltid bort den tillfälliga filen så att projektmappen inte fylls med temporära filer.
    cleanup_file(tmp_path)
