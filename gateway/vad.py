"""Silero VAD v5 segmenter — detects speech boundaries and yields audio segments.

Three modes (VAD_MODE env): `none` (fixed windows, no VAD), `batched`, `streaming`.
"""

import io
import logging
import struct
import wave
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from gateway.audio_source import AudioSource

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit PCM


def _pcm_to_wav(pcm_bytes: bytes) -> bytes:
    """Wrap raw PCM16 mono 16 kHz bytes into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class FixedWindowSegmenter:
    """No-VAD baseline: emit fixed-duration contiguous windows from the audio stream.

    Used for the control experiment "how much does VAD actually buy us?"
    """

    def __init__(self, window_ms: int = 5000) -> None:
        self._window_samples = SAMPLE_RATE * window_ms // 1000
        self._window_bytes = self._window_samples * SAMPLE_WIDTH

    def segments(self, source: "AudioSource") -> Iterator[tuple[int, str, bytes]]:
        buf = bytearray()
        seq = 0
        cursor_samples = 0
        for frame in source.frames():
            buf.extend(frame)
            while len(buf) >= self._window_bytes:
                window_pcm = bytes(buf[: self._window_bytes])
                buf = buf[self._window_bytes :]
                start_s = round(cursor_samples / SAMPLE_RATE, 1)
                end_s = round((cursor_samples + self._window_samples) / SAMPLE_RATE, 1)
                chunk_id = f"chunk_{start_s}s_{end_s}s"
                logging.info("Fixed-window segment #%d: %s", seq, chunk_id)
                yield seq, chunk_id, _pcm_to_wav(window_pcm)
                seq += 1
                cursor_samples += self._window_samples

        if buf:
            tail_samples = len(buf) // SAMPLE_WIDTH
            start_s = round(cursor_samples / SAMPLE_RATE, 1)
            end_s = round((cursor_samples + tail_samples) / SAMPLE_RATE, 1)
            chunk_id = f"chunk_{start_s}s_{end_s}s"
            logging.info("Fixed-window tail #%d: %s (%.1f s)", seq, chunk_id, end_s - start_s)
            yield seq, chunk_id, _pcm_to_wav(bytes(buf))


class SileroSegmenter:
    """Consume PCM frames from an AudioSource, run Silero VAD, yield speech segments.

    Two modes:
      * batched (default) — buffer the full stream, run get_speech_timestamps once.
      * streaming — VADIterator emits segments incrementally as silence boundaries arrive.

    Long segments (> ``max_segment_ms``) are split evenly into sub-segments so that
    downstream CTC ASR doesn't blow its context window.
    """

    def __init__(
        self,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        max_segment_ms: int = 20000,
        streaming: bool = False,
        speech_pad_ms: int = 100,
    ) -> None:
        self._streaming = streaming
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._max_segment_samples = SAMPLE_RATE * max_segment_ms // 1000
        self._speech_pad_ms = speech_pad_ms

        from silero_vad import get_speech_timestamps, load_silero_vad

        logging.info("Loading Silero VAD model...")
        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        logging.info("Silero VAD model loaded")

    def segments(self, source: "AudioSource") -> Iterator[tuple[int, str, bytes]]:
        """Yield ``(seq_num, chunk_id, wav_bytes)`` for each speech segment.

        ``seq_num`` is a zero-based sequential index.
        ``chunk_id`` encodes the time range, e.g. ``chunk_1.5s_4.2s``.
        ``wav_bytes`` is the segment audio in WAV format.
        """
        if self._streaming:
            yield from self._streaming_vad_segments(source)
        else:
            yield from self._vad_segments(source)

    # ------------------------------------------------------------------ #
    #  Batched VAD: buffer everything, single get_speech_timestamps call  #
    # ------------------------------------------------------------------ #
    def _vad_segments(self, source: "AudioSource") -> Iterator[tuple[int, str, bytes]]:
        import torch

        all_pcm = bytearray()
        for frame in source.frames():
            all_pcm.extend(frame)

        logging.info(
            "VAD: buffered %d bytes (%.1f s)",
            len(all_pcm), len(all_pcm) / SAMPLE_RATE / SAMPLE_WIDTH,
        )

        samples = struct.unpack(f"<{len(all_pcm) // SAMPLE_WIDTH}h", bytes(all_pcm))
        tensor = torch.FloatTensor(samples) / 32768.0

        timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=self._min_speech_ms,
            min_silence_duration_ms=self._min_silence_ms,
        )

        logging.info("VAD: detected %d speech segments", len(timestamps))

        raw_bytes = bytes(all_pcm)

        split_timestamps: list[dict[str, int]] = []
        for ts in timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            duration = end_sample - start_sample

            if duration <= self._max_segment_samples:
                split_timestamps.append(ts)
                continue

            n_parts = (duration + self._max_segment_samples - 1) // self._max_segment_samples
            part_len = duration // n_parts
            for i in range(n_parts):
                sub_start = start_sample + i * part_len
                sub_end = start_sample + (i + 1) * part_len if i < n_parts - 1 else end_sample
                split_timestamps.append({"start": sub_start, "end": sub_end})
            logging.info(
                "VAD: split long segment %.1fs-%.1fs into %d parts",
                start_sample / SAMPLE_RATE, end_sample / SAMPLE_RATE, n_parts,
            )

        logging.info(
            "VAD: %d segments after splitting (was %d)",
            len(split_timestamps), len(timestamps),
        )

        for seq, ts in enumerate(split_timestamps):
            s_start = ts["start"]
            s_end = ts["end"]
            segment_pcm = raw_bytes[s_start * SAMPLE_WIDTH : s_end * SAMPLE_WIDTH]

            start_s = round(s_start / SAMPLE_RATE, 1)
            end_s = round(s_end / SAMPLE_RATE, 1)
            chunk_id = f"chunk_{start_s}s_{end_s}s"

            logging.info("VAD segment #%d: %s (%.1f s)", seq, chunk_id, end_s - start_s)
            yield seq, chunk_id, _pcm_to_wav(segment_pcm)

    # ------------------------------------------------------------------ #
    #  Streaming VAD: VADIterator emits per silence boundary              #
    # ------------------------------------------------------------------ #
    def _streaming_vad_segments(self, source: "AudioSource") -> Iterator[tuple[int, str, bytes]]:
        import numpy as np
        import torch
        from silero_vad import VADIterator

        window_samples = 512
        window_bytes = window_samples * SAMPLE_WIDTH

        iterator = VADIterator(
            self._model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=self._min_silence_ms,
            speech_pad_ms=self._speech_pad_ms,
        )

        byte_buf = bytearray()
        collecting = False
        seg_pcm = bytearray()
        seg_start_sample = 0
        seq = 0

        def emit(start_sample: int, end_sample: int, pcm: bytes) -> Iterator[tuple[int, str, bytes]]:
            nonlocal seq
            duration = end_sample - start_sample
            if duration <= 0 or not pcm:
                return
            if duration <= self._max_segment_samples:
                start_s = round(start_sample / SAMPLE_RATE, 1)
                end_s = round(end_sample / SAMPLE_RATE, 1)
                chunk_id = f"chunk_{start_s}s_{end_s}s"
                logging.info("Streaming VAD segment #%d: %s (%.1f s)", seq, chunk_id, end_s - start_s)
                yield seq, chunk_id, _pcm_to_wav(pcm)
                seq += 1
                return

            n_parts = (duration + self._max_segment_samples - 1) // self._max_segment_samples
            part_samples = duration // n_parts
            for i in range(n_parts):
                sub_start = start_sample + i * part_samples
                sub_end = start_sample + (i + 1) * part_samples if i < n_parts - 1 else end_sample
                sub_pcm = pcm[(sub_start - start_sample) * SAMPLE_WIDTH : (sub_end - start_sample) * SAMPLE_WIDTH]
                start_s = round(sub_start / SAMPLE_RATE, 1)
                end_s = round(sub_end / SAMPLE_RATE, 1)
                chunk_id = f"chunk_{start_s}s_{end_s}s"
                logging.info("Streaming VAD sub-segment #%d: %s", seq, chunk_id)
                yield seq, chunk_id, _pcm_to_wav(sub_pcm)
                seq += 1

        for frame in source.frames():
            byte_buf.extend(frame)

            while len(byte_buf) >= window_bytes:
                window = bytes(byte_buf[:window_bytes])
                byte_buf = byte_buf[window_bytes:]

                if collecting:
                    seg_pcm.extend(window)

                arr = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
                event = iterator(torch.from_numpy(arr), return_seconds=False)

                if event:
                    if "start" in event:
                        collecting = True
                        seg_start_sample = event["start"]
                        seg_pcm = bytearray(window)
                    if "end" in event and collecting:
                        yield from emit(seg_start_sample, event["end"], bytes(seg_pcm))
                        collecting = False
                        seg_pcm = bytearray()

                # Force-split very long speech without silence
                if collecting and len(seg_pcm) >= self._max_segment_samples * SAMPLE_WIDTH:
                    force_end = seg_start_sample + self._max_segment_samples
                    yield from emit(seg_start_sample, force_end, bytes(seg_pcm))
                    seg_start_sample = force_end
                    seg_pcm = bytearray()

        # Stream ended — flush tail
        if collecting and seg_pcm:
            tail_end = seg_start_sample + len(seg_pcm) // SAMPLE_WIDTH
            yield from emit(seg_start_sample, tail_end, bytes(seg_pcm))

        iterator.reset_states()
