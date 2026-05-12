"""ASR Worker — listens to tasks:asr and runs T-one (t-tech/T-one, Conformer-CTC, CPU).

Note: T-one is tuned for 8 kHz telephony. Our gateway emits 16 kHz studio audio.
The library handles resampling internally via its `read_audio` helper, so we pass
the WAV file path directly. WER on 16 kHz domain audio is expected to be worse
than GigaAM-v3 — this comparison is informative, not a fair head-to-head.

Mutually exclusive with the GigaAM asr_worker — both consume `tasks:asr`.
"""

import base64
import json
import logging
import os
import tempfile
import time

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOG_DIR = os.getenv("LOG_DIR", "logs")

TASK_QUEUE = "tasks:asr"
RESULT_QUEUE = "results:asr"

pipeline = None


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "asr_worker_tone.log"), encoding="utf-8"),
        ],
    )


def connect_redis() -> redis.Redis:
    logging.info("Connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    logging.info("Redis connection OK")
    return r


def init_model() -> None:
    global pipeline
    from tone import StreamingCTCPipeline

    logging.info("Loading T-one StreamingCTCPipeline on cpu...")
    pipeline = StreamingCTCPipeline.from_hugging_face()
    logging.info("T-one pipeline loaded")


def transcribe_audio(audio_bytes: bytes) -> str:
    from tone import read_audio

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        audio = read_audio(tmp_path)
        phrases = pipeline.forward_offline(audio)
        text = " ".join(p.text for p in phrases).strip()
    finally:
        os.unlink(tmp_path)

    return text


def process_task(task: dict) -> dict:
    chunk_id = task.get("chunk_id", "unknown")
    audio_bytes = base64.b64decode(task.get("audio_b64", ""))

    t0 = time.perf_counter()
    text = transcribe_audio(audio_bytes)
    return {
        "chunk_id": chunk_id,
        "seq_num": task.get("seq_num"),
        "text": text,
        "language": "ru",
        "processing_time_s": round(time.perf_counter() - t0, 4),
    }


def main() -> None:
    setup_logging()
    logging.info("T-one ASR Worker starting")

    init_model()

    r = connect_redis()
    logging.info("Listening on queue: %s", TASK_QUEUE)

    while True:
        try:
            _, raw = r.blpop(TASK_QUEUE)
        except redis.ConnectionError as e:
            logging.error("Redis connection lost: %s. Retrying in 3s...", e)
            time.sleep(3)
            r = connect_redis()
            continue

        try:
            task = json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error("Bad JSON in task: %s", e)
            continue

        chunk_id = task.get("chunk_id", "unknown")
        seq_num = task.get("seq_num")
        logging.info("Received task: chunk_id=%s", chunk_id)

        try:
            result = process_task(task)
        except Exception as e:
            logging.error("Error processing chunk %s: %s", chunk_id, e)
            result = {
                "chunk_id": chunk_id,
                "seq_num": seq_num,
                "text": "",
                "language": "ru",
                "processing_time_s": 0.0,
                "error": str(e),
            }

        try:
            r.rpush(RESULT_QUEUE, json.dumps(result))
            logging.info("Result pushed: chunk_id=%s", chunk_id)
        except redis.ConnectionError as e:
            logging.error("Redis push failed (%s); reconnecting", e)
            time.sleep(3)
            r = connect_redis()


if __name__ == "__main__":
    main()
