"""File-based audio source — reads MP3/WAV and yields PCM frames."""

import logging
import time
from typing import Iterator

from pydub import AudioSegment

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit PCM


class FileAudioSource:
    """Load an audio file and yield fixed-size PCM16 mono 16 kHz frames.

    Parameters
    ----------
    filepath:
        Path to any audio file supported by pydub/ffmpeg.
    realtime_factor:
        Controls playback speed simulation.
        0.0 = as fast as possible (default, for testing).
        1.0 = real-time pace.
        0.1 = 10x faster than real-time.
    """

    sample_rate: int = SAMPLE_RATE

    def __init__(self, filepath: str, realtime_factor: float = 0.0) -> None:
        self._filepath = filepath
        self._realtime_factor = realtime_factor

    def frames(self, frame_duration_ms: int = 32) -> Iterator[bytes]:
        logging.info("Loading audio file: %s", self._filepath)
        audio = AudioSegment.from_file(self._filepath)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH)
        logging.info("Audio loaded: %d ms, converted to 16kHz mono PCM16", len(audio))

        raw = audio.raw_data
        frame_bytes = SAMPLE_WIDTH * (SAMPLE_RATE * frame_duration_ms // 1000)
        total_frames = (len(raw) + frame_bytes - 1) // frame_bytes

        logging.info(
            "Streaming %d frames (%d ms each, realtime_factor=%.2f)",
            total_frames,
            frame_duration_ms,
            self._realtime_factor,
        )

        sleep_time = frame_duration_ms / 1000.0 * self._realtime_factor

        for offset in range(0, len(raw), frame_bytes):
            chunk = raw[offset : offset + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk += b"\x00" * (frame_bytes - len(chunk))
            yield chunk
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self) -> None:
        pass
