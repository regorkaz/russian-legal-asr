"""Audio source protocol — abstraction for file / WebSocket / microphone input."""

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class AudioSource(Protocol):
    """Yields raw PCM16 mono 16 kHz frames."""

    sample_rate: int  # always 16000

    def frames(self, frame_duration_ms: int = 32) -> Iterator[bytes]:
        """Yield fixed-size PCM16 frames (``frame_duration_ms`` each)."""
        ...

    def close(self) -> None: ...
