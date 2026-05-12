"""AudioSource that pulls PCM16 frames from an in-process queue fed by a WebSocket."""

import logging
import queue
import threading
from typing import Iterator


class WebSocketAudioSource:
    """Thread-safe audio source. WebSocket handler pushes raw PCM16 mono 16 kHz
    bytes via :meth:`push`; the pipeline thread consumes them through :meth:`frames`.

    Frames are re-chunked to a fixed ``frame_duration_ms`` (default 32 ms, i.e. 512
    samples at 16 kHz), which is what the streaming VAD expects.
    """

    sample_rate: int = 16000

    def __init__(self, max_queue_bytes: int = 1024 * 1024) -> None:
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._closed = threading.Event()
        self._recorded = bytearray()  # full session PCM, for persistence
        self._recorded_lock = threading.Lock()

    def push(self, pcm_bytes: bytes) -> None:
        if self._closed.is_set():
            return
        with self._recorded_lock:
            self._recorded.extend(pcm_bytes)
        self._q.put(pcm_bytes)

    def close(self) -> None:
        self._closed.set()
        self._q.put(None)

    def frames(self, frame_duration_ms: int = 32) -> Iterator[bytes]:
        frame_bytes = 2 * (self.sample_rate * frame_duration_ms // 1000)
        buffer = bytearray()

        while True:
            try:
                item = self._q.get(timeout=30.0)
            except queue.Empty:
                logging.warning("WebSocketAudioSource: idle >30s, closing")
                return

            if item is None:
                break

            buffer.extend(item)
            while len(buffer) >= frame_bytes:
                yield bytes(buffer[:frame_bytes])
                buffer = buffer[frame_bytes:]

        # Flush tail (pad with zeros to frame size)
        if len(buffer) > 0:
            padded = bytes(buffer) + b"\x00" * (frame_bytes - len(buffer))
            yield padded

    def recorded_pcm(self) -> bytes:
        with self._recorded_lock:
            return bytes(self._recorded)
