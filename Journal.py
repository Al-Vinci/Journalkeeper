import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer

from audio import cleanup_file, clear_saved_wavs, list_saved_wavs, save_temp_audio
from audio_stream import AudioProcessor
from generate import generate_journal
from stream_transcribe import text_queue
from transcribe import transcribe_audio


st.set_page_config(page_title="Vet Journal AI", layout="wide")

st.title("Veterinarjournal - AI-utkast")
st.warning("Detta ar ett UTKAST. Maste granskas av veterinär innan anvandning.")

st.subheader("Live transkribering")

if "echo_cancellation" not in st.session_state:
    st.session_state.echo_cancellation = False
if "noise_suppression" not in st.session_state:
    st.session_state.noise_suppression = False
if "auto_gain_control" not in st.session_state:
    st.session_state.auto_gain_control = False
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

with st.expander("Mikrofoninstallningar", expanded=True):
    st.checkbox("Echo cancellation", key="echo_cancellation")
    st.checkbox("Noise suppression", key="noise_suppression")
    st.checkbox("Auto gain control", key="auto_gain_control")
    st.caption("Om ljudet blir avhugget eller pumpande: lat alla tre vara avstangda.")

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

output = st.empty()

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
    output.markdown(st.session_state.full_text)

st.subheader("Sparade backupfiler")

if st.button("Rensa sparade WAV-filer"):
    deleted_files = clear_saved_wavs()
    st.success(f"Raderade {deleted_files} sparade WAV-filer.")
    st.rerun()

saved_wavs = list_saved_wavs()

if saved_wavs:
    for wav_path in saved_wavs:
        st.write(wav_path.name)
        st.audio(str(wav_path), format="audio/wav")
else:
    st.info("Inga sparade WAV-filer an.")

audio_file = st.file_uploader("Ladda upp ljud (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

if audio_file is None:
    st.info("Ladda upp en ljudfil for att borja")
    st.stop()

file_extension = audio_file.name.split(".")[-1].lower()

if file_extension not in ["wav", "mp3", "m4a"]:
    st.error("Endast wav/mp3/m4a stods")
    st.stop()

tmp_path = save_temp_audio(audio_file.read(), file_extension)

try:
    st.info("Transkriberar...")
    text = transcribe_audio(tmp_path)

    st.subheader("Transkribering")
    st.write(text)

    st.info("Skapar journalutkast...")
    journal = generate_journal(text)

    st.subheader("Journalutkast (SOAP)")
    st.text_area("Redigera vid behov", journal, height=300)

    st.success("Klart! Kopiera texten manuellt till journalsystemet.")
finally:
    cleanup_file(tmp_path)
