"""Streaming pipeline — 3-thread orchestration for real-time ASR + speaker ID."""

import base64
import json
import logging
import queue
import threading
import time

import redis

from gateway.audio_source import AudioSource
from gateway.result_collector import OrderedTranscriptWriter
from gateway.vad import FixedWindowSegmenter, SileroSegmenter

# Sentinel value to signal end of stream
_SENTINEL = None


class StreamingPipeline:
    """Three-thread streaming pipeline.

    Threads
    -------
    Main thread (caller):
        AudioSource -> SileroSegmenter -> send_queue
    Sender thread:
        send_queue -> Redis (tasks:asr + tasks:speaker)
    Collector thread:
        Redis (results:asr + results:speaker) -> OrderedTranscriptWriter
    """

    def __init__(
        self,
        source: AudioSource,
        redis_client: redis.Redis,
        output_path: str | None,
        timings_path: str | None = None,
        vad_mode: str = "streaming",
        vad_kwargs: dict | None = None,
        on_line=None,
        meta: dict | None = None,
    ) -> None:
        if vad_mode not in ("none", "batched", "streaming"):
            raise ValueError(f"vad_mode must be none/batched/streaming, got {vad_mode!r}")
        self._source = source
        self._redis = redis_client
        self._output_path = output_path
        self._timings_path = timings_path
        self._vad_mode = vad_mode
        self._vad_kwargs = vad_kwargs or {}
        self._on_line = on_line
        self._meta = meta or {}

        self._send_queue: queue.Queue[tuple[int, str, bytes] | None] = queue.Queue(maxsize=10)
        self._chunks_sent = 0
        self._lock = threading.Lock()
        self._done_sending = threading.Event()
        self._error: Exception | None = None
        self._writer: OrderedTranscriptWriter | None = None

    def run(self) -> None:
        """Run the full pipeline. Blocks until all results are collected."""
        wall_start = time.perf_counter()

        self._writer = OrderedTranscriptWriter(
            self._output_path,
            self._timings_path,
            on_line=self._on_line,
            meta=self._meta,
        )

        collector_thread = threading.Thread(
            target=self._collect_results, daemon=True, name="collector"
        )
        sender_thread = threading.Thread(
            target=self._send_chunks, daemon=True, name="sender"
        )

        collector_thread.start()
        sender_thread.start()

        # Main thread: produce segments
        try:
            if self._vad_mode == "none":
                segmenter = FixedWindowSegmenter()
            else:
                segmenter = SileroSegmenter(
                    streaming=(self._vad_mode == "streaming"),
                    **self._vad_kwargs,
                )
            for seq, chunk_id, wav_bytes in segmenter.segments(self._source):
                self._send_queue.put((seq, chunk_id, wav_bytes))
                logging.info("Produced segment: seq=%d chunk_id=%s", seq, chunk_id)
        except Exception as e:
            logging.error("Error in segmenter: %s", e)
            self._error = e
        finally:
            self._send_queue.put(_SENTINEL)

        sender_thread.join()
        collector_thread.join()

        wall_clock = time.perf_counter() - wall_start
        self._writer.write_timings(wall_clock_s=wall_clock)
        self._writer.close()

        if self._error:
            logging.error("Pipeline finished with error: %s", self._error)
            raise self._error

        logging.info(
            "Pipeline complete: %d chunks sent, %d lines written, %d buffered (orphan), wall=%.2fs",
            self._chunks_sent,
            self._writer.written_count,
            self._writer.buffered_count,
            wall_clock,
        )

    # ------------------------------------------------------------------ #
    #  Sender thread                                                      #
    # ------------------------------------------------------------------ #
    def _send_chunks(self) -> None:
        try:
            while True:
                item = self._send_queue.get()
                if item is _SENTINEL:
                    logging.info("Sender: end of stream, %d chunks sent", self._chunks_sent)
                    self._done_sending.set()
                    return

                seq, chunk_id, wav_bytes = item
                encoded = base64.b64encode(wav_bytes).decode("ascii")
                task = json.dumps({
                    "chunk_id": chunk_id,
                    "seq_num": seq,
                    "audio_b64": encoded,
                })
                sent_at = time.perf_counter()
                self._redis.rpush("tasks:asr", task)
                self._redis.rpush("tasks:speaker", task)
                if self._writer is not None:
                    self._writer.record_sent(seq, sent_at)

                with self._lock:
                    self._chunks_sent += 1

                logging.info("Sent chunk seq=%d (%s) to workers", seq, chunk_id)

        except Exception as e:
            logging.error("Sender thread error: %s", e)
            self._error = e
            self._done_sending.set()

    # ------------------------------------------------------------------ #
    #  Collector thread                                                   #
    # ------------------------------------------------------------------ #
    def _collect_results(self) -> None:
        writer = self._writer
        assert writer is not None
        asr_received = 0
        speaker_received = 0

        try:
            while True:
                # Check termination: all results collected
                with self._lock:
                    expected = self._chunks_sent

                if self._done_sending.is_set() and asr_received >= expected and speaker_received >= expected:
                    logging.info(
                        "Collector: all results received (asr=%d, speaker=%d)",
                        asr_received,
                        speaker_received,
                    )
                    return

                # blpop on both result queues with short timeout
                resp = self._redis.blpop(
                    ["results:asr", "results:speaker"], timeout=2
                )

                if resp is None:
                    # Timeout — check if we should stop
                    if self._done_sending.is_set():
                        with self._lock:
                            expected = self._chunks_sent
                        if asr_received >= expected and speaker_received >= expected:
                            return
                    continue

                queue_name, raw = resp
                # redis-py may return bytes or str depending on decode_responses
                if isinstance(queue_name, bytes):
                    queue_name = queue_name.decode()

                result = json.loads(raw)

                if queue_name == "results:asr":
                    asr_received += 1
                    writer.add_asr_result(result)
                    logging.info(
                        "Collected ASR result: seq=%s chunk=%s (%d/%d)",
                        result.get("seq_num"),
                        result.get("chunk_id"),
                        asr_received,
                        expected,
                    )
                elif queue_name == "results:speaker":
                    speaker_received += 1
                    writer.add_speaker_result(result)
                    logging.info(
                        "Collected Speaker result: seq=%s chunk=%s (%d/%d)",
                        result.get("seq_num"),
                        result.get("chunk_id"),
                        speaker_received,
                        expected,
                    )

        except Exception as e:
            logging.error("Collector thread error: %s", e)
            self._error = e
