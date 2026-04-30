from diarization import transcribe_audio_with_diarization


# Tar en ljudfil från disk och returnerar både vanlig text och diariserad text.
def transcribe_audio(file_path):
    return transcribe_audio_with_diarization(file_path)
