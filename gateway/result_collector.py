"""Ordered transcript writer — buffers results and flushes in seq_num order."""

import json
import logging
import os
import re
import time
from dataclasses import dataclass


_CHUNK_ID_RE = re.compile(r"chunk_(?P<start>[\d.]+)s_(?P<end>[\d.]+)s")


def _parse_chunk_id(chunk_id: str) -> tuple[float | None, float | None]:
    m = _CHUNK_ID_RE.search(chunk_id or "")
    if not m:
        return None, None
    return float(m.group("start")), float(m.group("end"))


@dataclass
class _PendingResult:
    asr: dict | None = None
    speaker: dict | None = None

    def is_complete(self) -> bool:
        return self.asr is not None and self.speaker is not None


class OrderedTranscriptWriter:
    """Buffer ASR and speaker results; write transcript lines in strict seq_num order.

    Also aggregates per-chunk timing: audio duration, ASR/Speaker processing time,
    end-to-end latency (sent → written). Persisted via ``write_timings``.
    """

    def __init__(
        self,
        output_path: str | None,
        timings_path: str | None = None,
        on_line=None,
        meta: dict | None = None,
    ) -> None:
        self._output_path = output_path
        self._timings_path = timings_path
        self._on_line = on_line
        self._meta = meta or {}
        self._buffer: dict[int, _PendingResult] = {}
        self._next_seq: int = 0
        self._file = None

        self._sent_times: dict[int, float] = {}
        self._asr_arrived_at: dict[int, float] = {}
        self._speaker_arrived_at: dict[int, float] = {}
        self._timings: list[dict] = []

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            self._file = open(output_path, "w", encoding="utf-8")

    def record_sent(self, seq: int, sent_time: float) -> None:
        self._sent_times[seq] = sent_time

    def add_asr_result(self, result: dict) -> list[str]:
        seq = result.get("seq_num", -1)
        if seq not in self._buffer:
            self._buffer[seq] = _PendingResult()
        self._buffer[seq].asr = result
        self._asr_arrived_at[seq] = time.perf_counter()
        return self._try_flush()

    def add_speaker_result(self, result: dict) -> list[str]:
        seq = result.get("seq_num", -1)
        if seq not in self._buffer:
            self._buffer[seq] = _PendingResult()
        self._buffer[seq].speaker = result
        self._speaker_arrived_at[seq] = time.perf_counter()
        return self._try_flush()

    def _try_flush(self) -> list[str]:
        """Flush all consecutive complete results starting from ``_next_seq``."""
        flushed: list[str] = []
        while self._next_seq in self._buffer and self._buffer[self._next_seq].is_complete():
            pending = self._buffer.pop(self._next_seq)
            line = self._format_line(pending)
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()
            self._record_timing(self._next_seq, pending)
            flushed.append(line)
            logging.info("Transcript [seq=%d]: %s", self._next_seq, line)

            if self._on_line is not None:
                chunk_id = pending.asr.get("chunk_id", "")
                start_s, end_s = _parse_chunk_id(chunk_id)
                try:
                    self._on_line({
                        "seq": self._next_seq,
                        "chunk_id": chunk_id,
                        "start_s": start_s,
                        "end_s": end_s,
                        "speaker": pending.speaker.get("speaker", "Unknown"),
                        "speaker_confidence": pending.speaker.get("confidence"),
                        "text": pending.asr.get("text", ""),
                    })
                except Exception as e:
                    logging.warning("on_line callback failed: %s", e)

            self._next_seq += 1
        return flushed

    def _record_timing(self, seq: int, pending: _PendingResult) -> None:
        chunk_id = (pending.asr or {}).get("chunk_id") or (pending.speaker or {}).get("chunk_id", "")
        start_s, end_s = _parse_chunk_id(chunk_id)
        duration = (end_s - start_s) if (start_s is not None and end_s is not None) else None

        sent_t = self._sent_times.pop(seq, None)
        asr_t = self._asr_arrived_at.pop(seq, None)
        spk_t = self._speaker_arrived_at.pop(seq, None)
        final_t = time.perf_counter()

        asr_latency = (asr_t - sent_t) if (sent_t is not None and asr_t is not None) else None
        id_latency = (spk_t - sent_t) if (sent_t is not None and spk_t is not None) else None
        final_latency = (final_t - sent_t) if sent_t is not None else None

        self._timings.append({
            "seq": seq,
            "chunk_id": chunk_id,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration,
            "asr_processing_s": (pending.asr or {}).get("processing_time_s"),
            "speaker_processing_s": (pending.speaker or {}).get("processing_time_s"),
            "asr_latency_s": asr_latency,
            "id_latency_s": id_latency,
            "final_latency_s": final_latency,
            "speaker": (pending.speaker or {}).get("speaker"),
            "speaker_confidence": (pending.speaker or {}).get("confidence"),
        })

    @staticmethod
    def _format_line(pending: _PendingResult) -> str:
        chunk_id = pending.asr.get("chunk_id", "?")
        speaker = pending.speaker.get("speaker", "Unknown")
        text = pending.asr.get("text", "")
        return f"[{speaker}] ({chunk_id}): {text}"

    def write_timings(self, wall_clock_s: float) -> None:
        if not self._timings_path:
            return

        audio_duration = 0.0
        for t in self._timings:
            end_s = t.get("end_s")
            if end_s is not None and end_s > audio_duration:
                audio_duration = end_s

        payload = {
            "audio_duration_s": audio_duration,
            "wall_clock_s": wall_clock_s,
            "n_chunks": len(self._timings),
            **self._meta,
            "chunks": self._timings,
        }
        os.makedirs(os.path.dirname(self._timings_path) or ".", exist_ok=True)
        with open(self._timings_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info("Timings written: %s", self._timings_path)

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
            logging.info("Transcript file closed: %s", self._output_path)

    @property
    def written_count(self) -> int:
        return self._next_seq

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)
