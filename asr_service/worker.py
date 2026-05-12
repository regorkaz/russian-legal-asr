"""ASR Worker — listens to tasks:asr queue in Redis and runs GigaAM-v3 CTC.

Decoding path: encoder + CTC head → log-probs → manual greedy CTC collapse →
sentencepiece detokenize. This is the seam where KenLM shallow fusion /
hotword biasing will be plugged in (via pyctcdecode beam search on log_probs).
"""

import base64
import json
import logging
import os
import tempfile
import time

import redis
import torch

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LOG_DIR = os.getenv("LOG_DIR", "logs")

# LM_MODE = greedy | hotwords | kenlm | hotwords_kenlm
# greedy is the verified baseline; others go through pyctcdecode beam search.
LM_MODE = os.getenv("LM_MODE", "greedy")
LM_BEAM_WIDTH = int(os.getenv("LM_BEAM_WIDTH", "10"))
LM_ALPHA = float(os.getenv("LM_ALPHA", "0.5"))
LM_BETA = float(os.getenv("LM_BETA", "1.5"))
HOTWORDS_PATH = os.getenv("HOTWORDS_PATH", "/app/lexicon.txt")
HOTWORD_WEIGHT = float(os.getenv("HOTWORD_WEIGHT", "10.0"))
KENLM_PATH = os.getenv("KENLM_PATH", "/app/kenlm/legal.bin")
UNIGRAMS_PATH = os.getenv("UNIGRAMS_PATH", "/app/kenlm/unigrams.txt")

TASK_QUEUE = "tasks:asr"
RESULT_QUEUE = "results:asr"

asr_model = None
sp_tokenizer = None
blank_id = None
beam_decoder = None  # pyctcdecode decoder; None for LM_MODE=greedy
hotwords: list[str] = []


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, "asr_worker.log"), encoding="utf-8"),
        ],
    )


def connect_redis() -> redis.Redis:
    logging.info("Connecting to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    logging.info("Redis connection OK")
    return r


def init_model() -> None:
    global asr_model, sp_tokenizer, blank_id
    import gigaam

    logging.info("Loading GigaAM-v3 E2E CTC model on cpu...")
    try:
        asr_model = gigaam.load_model("v3_e2e_ctc", device="cpu", fp16_encoder=False)
    except AssertionError as e:
        # GigaAM verifies SHA-256 of the cached checkpoint and raises a bare
        # AssertionError on mismatch — happens when the first download is truncated.
        if "checksum" not in str(e).lower():
            raise
        ckpt = os.path.expanduser("~/.cache/gigaam/v3_e2e_ctc.ckpt")
        if os.path.exists(ckpt):
            logging.warning("GigaAM checksum failed; removing %s and retrying", ckpt)
            os.unlink(ckpt)
        asr_model = gigaam.load_model("v3_e2e_ctc", device="cpu", fp16_encoder=False)
    logging.info("ASR model loaded")

    sp_tokenizer = asr_model.decoding.tokenizer.model
    blank_id = len(asr_model.decoding.tokenizer)  # GigaAM appends blank at the end
    logging.info("CTC greedy decoder ready (vocab=%d, blank=%d)", blank_id, blank_id)

    if LM_MODE != "greedy":
        _init_beam_decoder()


def _init_beam_decoder() -> None:
    global beam_decoder, hotwords
    from pyctcdecode import build_ctcdecoder

    labels = [sp_tokenizer.IdToPiece(i) for i in range(blank_id)] + [""]

    kenlm_path = KENLM_PATH if LM_MODE in ("kenlm", "hotwords_kenlm") else None
    if kenlm_path and not os.path.isfile(kenlm_path):
        raise FileNotFoundError(f"LM_MODE={LM_MODE} but KenLM file not found: {kenlm_path}")

    unigrams: list[str] | None = None
    if os.path.isfile(UNIGRAMS_PATH):
        with open(UNIGRAMS_PATH, encoding="utf-8") as f:
            unigrams = [line.strip() for line in f if line.strip()]
        logging.info("Loaded %d unigrams from %s", len(unigrams), UNIGRAMS_PATH)
    else:
        logging.warning("UNIGRAMS_PATH not found: %s (decoding accuracy may degrade)", UNIGRAMS_PATH)

    beam_decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=kenlm_path,
        unigrams=unigrams,
        alpha=LM_ALPHA if kenlm_path else 0.0,
        beta=LM_BETA if kenlm_path else 0.0,
    )

    if LM_MODE in ("hotwords", "hotwords_kenlm"):
        if not os.path.isfile(HOTWORDS_PATH):
            raise FileNotFoundError(f"LM_MODE={LM_MODE} but hotwords file not found: {HOTWORDS_PATH}")
        with open(HOTWORDS_PATH, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    hotwords.append(s)
        logging.info("Loaded %d hotwords from %s", len(hotwords), HOTWORDS_PATH)

    logging.info(
        "Beam decoder ready: mode=%s beam=%d alpha=%.2f beta=%.2f hw_weight=%.1f kenlm=%s",
        LM_MODE, LM_BEAM_WIDTH, LM_ALPHA, LM_BETA, HOTWORD_WEIGHT, kenlm_path,
    )


def _ctc_greedy_collapse(frame_ids) -> list[int]:
    """Standard CTC: dedupe consecutive duplicates, then drop blanks."""
    collapsed: list[int] = []
    prev = -1
    for i in frame_ids:
        if i != prev:
            collapsed.append(int(i))
            prev = int(i)
    return [i for i in collapsed if i != blank_id]


def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        wav, length = asr_model.prepare_wav(tmp_path)
        with torch.inference_mode():
            encoded, encoded_len = asr_model.forward(wav, length)
            log_probs = asr_model.head(encoder_output=encoded)
        # log_probs: [B, T, V+1], already log-softmaxed.
        t_frames = int(encoded_len[0])
        slice_lp = log_probs[0, :t_frames]
    finally:
        os.unlink(tmp_path)

    if LM_MODE == "greedy":
        frame_ids = slice_lp.argmax(dim=-1).cpu().tolist()
        return sp_tokenizer.decode(_ctc_greedy_collapse(frame_ids))

    # pyctcdecode beam search (hotwords / kenlm / both).
    # NOTE: pyctcdecode owns detokenization here — its BPE join can add minor
    # whitespace artifacts before punctuation (see A/B sanity check), but for
    # biased/LM decoding we accept that as the cost of the seam; gains from
    # hotwords/KenLM dominate.
    lp_np = slice_lp.float().cpu().numpy()
    kwargs: dict = {"beam_width": LM_BEAM_WIDTH}
    if hotwords:
        kwargs["hotwords"] = hotwords
        kwargs["hotword_weight"] = HOTWORD_WEIGHT
    return beam_decoder.decode(lp_np, **kwargs)


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
    logging.info("ASR Worker starting")

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
