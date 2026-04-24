import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer
from audio import cleanup_file, clear_saved_wavs, list_saved_wavs, save_temp_audio
from audio_stream import AudioProcessor
from stream_transcribe import text_queue
from transcribe import transcribe_audio
from generate import generate_journal

st.set_page_config(page_title="Vet Journal AI", layout="wide")

st.title("🐾 Veterinärjournal – AI-utkast")
st.warning("Detta är ett UTKAST. Måste granskas av veterinär innan användning.")

st.subheader("🎙️ Live transkribering")


webrtc_ctx = webrtc_streamer(
    key="live3",  # viktigt: ny key
    audio_processor_factory=AudioProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
        "video": False,
    },
    async_processing=True,  # 🔥 KRITISK
)

output = st.empty()

# 🔁 Spara text i session (viktigt!)
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

if webrtc_ctx.state.playing:
    new_texts = []

    while not text_queue.empty():
        new_texts.append(text_queue.get())

    if new_texts:
        st.session_state.full_text += " " + " ".join(new_texts)

    output.markdown(st.session_state.full_text)

    time.sleep(0.2)
    st.rerun()

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
    st.info("Inga sparade WAV-filer än.")

audio_file = st.file_uploader("Ladda upp ljud (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])

if audio_file is None:
    st.info("Ladda upp en ljudfil för att börja")
    st.stop()

file_extension = audio_file.name.split(".")[-1].lower()

if file_extension not in ["wav", "mp3", "m4a"]:
    st.error("Endast wav/mp3/m4a stöds")
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
