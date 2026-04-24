from collections import deque
import io
import wave

import av
import numpy as np
import webrtcvad
from streamlit_webrtc import AudioProcessorBase

from audio import create_live_recording_path
from stream_transcribe import audio_queue


class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        print("AudioProcessor INIT")
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.samples_per_frame = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.bytes_per_frame = self.samples_per_frame * 2
        self.pre_roll_frames = max(1, 300 // self.frame_duration_ms)
        self.post_roll_frames = max(1, 450 // self.frame_duration_ms)
        self.overlap_frames = max(1, 180 // self.frame_duration_ms)
        self.max_chunk_frames = max(1, 15000 // self.frame_duration_ms)

        self.vad = webrtcvad.Vad(2)
        self.pending_pcm = bytearray()
        self.pre_roll_buffer = deque(maxlen=self.pre_roll_frames)
        self.overlap_seed_frames = []
        self.current_chunk_frames = []
        self.in_speech = False
        self.trailing_silence_frames = 0
        self.speech_frames_in_chunk = 0
        self.chunk_counter = 0

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
        print("recv_queued called, frames:", len(frames))

        for frame in frames:
            resampled_frames = self.resampler.resample(frame)
            for resampled_frame in resampled_frames:
                audio = resampled_frame.to_ndarray().reshape(-1).astype(np.int16)
                if audio.size == 0:
                    continue

                audio_bytes = audio.tobytes()
                self.recording_wav.writeframes(audio_bytes)
                self.recording_file.flush()
                self.process_pcm_bytes(audio_bytes)

        return frames

    def process_pcm_bytes(self, pcm_bytes):
        self.pending_pcm.extend(pcm_bytes)

        while len(self.pending_pcm) >= self.bytes_per_frame:
            frame_bytes = bytes(self.pending_pcm[: self.bytes_per_frame])
            del self.pending_pcm[: self.bytes_per_frame]
            self.process_vad_frame(frame_bytes)

    def process_vad_frame(self, frame_bytes):
        is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
        self.pre_roll_buffer.append(frame_bytes)

        if not self.in_speech:
            if not is_speech:
                return

            self.in_speech = True
            self.trailing_silence_frames = 0
            self.current_chunk_frames = list(self.overlap_seed_frames)
            self.current_chunk_frames.extend(self.pre_roll_buffer)
            self.speech_frames_in_chunk = sum(
                1 for chunk_frame in self.current_chunk_frames if self.vad.is_speech(chunk_frame, self.sample_rate)
            )
            self.overlap_seed_frames = []
            self.pre_roll_buffer.clear()
            print("Speech started")
            return

        self.current_chunk_frames.append(frame_bytes)

        if is_speech:
            self.speech_frames_in_chunk += 1
            self.trailing_silence_frames = 0
        else:
            self.trailing_silence_frames += 1

        if len(self.current_chunk_frames) >= self.max_chunk_frames:
            print("Finalizing chunk at max length")
            self.finalize_chunk(continue_speech=True)
            return

        if self.trailing_silence_frames >= self.post_roll_frames:
            print("Finalizing chunk on pause")
            self.finalize_chunk(continue_speech=False)

    def finalize_chunk(self, continue_speech):
        chunk_frames = self.current_chunk_frames
        self.current_chunk_frames = []
        self.trailing_silence_frames = 0

        if not chunk_frames or self.speech_frames_in_chunk == 0:
            self.speech_frames_in_chunk = 0
            self.in_speech = continue_speech
            if not continue_speech:
                self.pre_roll_buffer.clear()
                self.overlap_seed_frames = []
            return

        self.overlap_seed_frames = chunk_frames[-self.overlap_frames :]
        audio_bytes = b"".join(chunk_frames)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        self.chunk_counter += 1
        self.send_chunk(audio_data)

        if continue_speech:
            self.current_chunk_frames = list(self.overlap_seed_frames)
            self.speech_frames_in_chunk = sum(
                1 for chunk_frame in self.current_chunk_frames if self.vad.is_speech(chunk_frame, self.sample_rate)
            )
            self.in_speech = True
        else:
            self.speech_frames_in_chunk = 0
            self.in_speech = False
            self.pre_roll_buffer.clear()

    def send_chunk(self, audio_data):
        print("Sending chunk!")
        if audio_data.size == 0:
            return

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        wav_io.seek(0)
        wav_io.name = f"live_chunk_{self.chunk_counter}.wav"
        audio_queue.put(wav_io)

    def on_ended(self):
        if self.in_speech and self.current_chunk_frames:
            print("Finalizing chunk on stream end")
            self.finalize_chunk(continue_speech=False)
        self.recording_wav.close()
        self.recording_file.close()
