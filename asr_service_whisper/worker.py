"""ASR Worker — listens to tasks:asr in Redis and runs Whisper-large-v3 (faster-whisper, CPU int8).

Mutually exclusive with the GigaAM asr_worker — both consume the same `tasks:asr`
queue, so only one container runs at a time. Selected via docker compose profile.
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
MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))

TASK_QUEUE = "tasks:asr"
RESULT_QUEUE = "results:asr"

asr_model = None


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "asr_worker_whisper.log"), encoding="utf-8"),
        ],
    )


def connect_redis() -> redis.Redis:
    logging.info("Connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    logging.info("Redis connection OK")
    return r


def init_model() -> None:
    global asr_model
    from faster_whisper import WhisperModel

    logging.info("Loading Whisper %s on cpu (compute_type=%s)...", MODEL_NAME, COMPUTE_TYPE)
    asr_model = WhisperModel(MODEL_NAME, device="cpu", compute_type=COMPUTE_TYPE)
    logging.info("Whisper model loaded")


def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, _ = asr_model.transcribe(
            tmp_path,
            language="ru",
            beam_size=BEAM_SIZE,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
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
    logging.info("Whisper ASR Worker starting")

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
