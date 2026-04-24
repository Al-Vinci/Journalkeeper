import av
import numpy as np
from streamlit_webrtc import AudioProcessorBase
from audio import create_live_recording_path
from stream_transcribe import audio_queue
import io
import wave

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        print("🔥 AudioProcessor INIT")
        self.buffer = []
        self.sample_rate = 16000
        self.buffered_samples = 0
        self.chunk_samples = self.sample_rate * 2
        self.min_rms = 0.003
        self.min_peak = 0.02
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.sample_rate,
        )
        self.recording_path = create_live_recording_path()
        self.recording_file = open(self.recording_path, "wb")
        self.recording_wav = wave.open(self.recording_file, "wb")
        self.recording_wav.setnchannels(1)
        self.recording_wav.setsampwidth(2)
        self.recording_wav.setframerate(self.sample_rate)

    async def recv_queued(self, frames):
        print("🎙️ recv_queued called, frames:", len(frames))

        for frame in frames:
            resampled_frames = self.resampler.resample(frame)
            for resampled_frame in resampled_frames:
                audio = resampled_frame.to_ndarray().reshape(-1).astype(np.int16)
                if audio.size == 0:
                    continue

                self.buffer.append(audio)
                self.buffered_samples += audio.size
                self.recording_wav.writeframes(audio.tobytes())
                self.recording_file.flush()

        if self.buffered_samples >= self.chunk_samples:
            print("📦 Trigger send_chunk")
            self.send_chunk()

        return frames

    def chunk_has_speech(self, audio_data):
        if audio_data.size == 0:
            return False

        normalized = audio_data.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(normalized))))
        peak = float(np.max(np.abs(normalized)))
        print(f"🔉 Chunk stats rms={rms:.5f} peak={peak:.5f}")
        return rms >= self.min_rms or peak >= self.min_peak

    def send_chunk(self):
        print("🚀 Sending chunk!")
        if not self.buffer:
            return

        print("🚀 Sending chunk!")

        audio_data = np.concatenate(self.buffer)
        self.buffer = []
        self.buffered_samples = 0

        if not self.chunk_has_speech(audio_data):
            print("🔇 Skipping silent chunk")
            return

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        wav_io.seek(0)
        wav_io.name = "live_chunk.wav"
        audio_queue.put(wav_io)

    def on_ended(self):
        self.send_chunk()
        self.recording_wav.close()
        self.recording_file.close()
