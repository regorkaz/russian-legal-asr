"""FastAPI web gateway: voice enrollment + live recording + mp3 upload, single workspace."""

import asyncio
import io
import json
import logging
import os
import queue
import threading
import time
import uuid
import wave
from pathlib import Path

import redis
from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment

from gateway.pipeline import StreamingPipeline
from gateway.sources.file_source import FileAudioSource
from webgateway.ws_audio_source import WebSocketAudioSource

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DATA_DIR = os.getenv("DATA_DIR", "./data")
WORKSPACE_DIR = os.path.join(DATA_DIR, "web_workspace")
VOICES_DIR = os.path.join(WORKSPACE_DIR, "voices")
AUDIO_PATH = os.path.join(WORKSPACE_DIR, "audio.wav")
TRANSCRIPT_PATH = os.path.join(WORKSPACE_DIR, "transcript.txt")
TIMINGS_PATH = os.path.join(WORKSPACE_DIR, "timings.json")
LOG_DIR = os.getenv("LOG_DIR", "logs")
CONSENT_LOG = os.path.join(LOG_DIR, "consent.log")

# Path to voices as seen by speaker_worker (it mounts ./data at /data).
SPEAKER_VOICES_REMOTE = os.getenv("SPEAKER_VOICES_REMOTE", "/data/web_workspace/voices")

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

SESSION_FINALIZE_TIMEOUT_S = 3.0

os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "webgateway.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("webgateway")

app = FastAPI(title="Legal-ASR Web Gateway")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------- #
#  Redis + active-operation lock                                          #
# ---------------------------------------------------------------------- #

def _redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


_op_lock = threading.Lock()
_active_op: str | None = None  # "live:<session_id>" | "upload:<task_id>" | None


def _acquire_op(tag: str) -> bool:
    global _active_op
    with _op_lock:
        if _active_op is not None:
            return False
        _active_op = tag
        return True


def _release_op(tag: str) -> None:
    global _active_op
    with _op_lock:
        if _active_op == tag:
            _active_op = None


def _force_release() -> str | None:
    global _active_op
    with _op_lock:
        prev = _active_op
        _active_op = None
        return prev


def _notify_speaker_reload() -> None:
    try:
        _redis().rpush(
            "control:speaker",
            json.dumps({"action": "reload", "enrollment_dir": SPEAKER_VOICES_REMOTE}),
        )
        log.info("Sent reload to speaker_worker (dir=%s)", SPEAKER_VOICES_REMOTE)
    except Exception as e:
        log.warning("Failed to notify speaker_worker: %s", e)


# ---------------------------------------------------------------------- #
#  Root + static                                                          #
# ---------------------------------------------------------------------- #

@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------- #
#  Voices (per-workspace)                                                 #
# ---------------------------------------------------------------------- #

def _sanitize_name(name: str) -> str:
    safe = "".join(c for c in name.strip() if c.isalnum() or c in ("_", "-"))
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid speaker name")
    return safe[:40]


@app.get("/voices")
def list_voices() -> dict:
    items = []
    for p in sorted(Path(VOICES_DIR).glob("*.mp3")):
        try:
            duration_ms = len(AudioSegment.from_file(p))
        except Exception:
            duration_ms = 0
        items.append({"label": p.stem, "duration_ms": duration_ms})
    return {"voices": items}


@app.put("/voices/{label}")
async def put_voice(label: str, file: UploadFile) -> dict:
    safe = _sanitize_name(label)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        audio = AudioSegment.from_file(io.BytesIO(data))
    except Exception as e:
        log.warning("Voice decode failed for %s: %s", safe, e)
        raise HTTPException(status_code=400, detail=f"Cannot decode audio: {e}")

    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH)
    out_path = os.path.join(VOICES_DIR, f"{safe}.mp3")
    audio.export(out_path, format="mp3", bitrate="96k")
    log.info("Voice saved: %s (%d ms)", out_path, len(audio))

    _notify_speaker_reload()
    return {"label": safe, "duration_ms": len(audio)}


@app.delete("/voices/{label}")
def delete_voice(label: str) -> dict:
    safe = _sanitize_name(label)
    p = Path(VOICES_DIR) / f"{safe}.mp3"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    p.unlink()
    _notify_speaker_reload()
    return {"removed": safe}


# ---------------------------------------------------------------------- #
#  Abort                                                                  #
# ---------------------------------------------------------------------- #

@app.post("/abort")
def abort_active() -> dict:
    prev = _force_release()
    log.info("Abort called; previous active op: %s", prev)
    return {"aborted": prev}


# ---------------------------------------------------------------------- #
#  Consent + persistence helpers                                          #
# ---------------------------------------------------------------------- #

def _log_consent(tag: str, user_agent: str, client_host: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{ts} op={tag} consent=true client={client_host} user_agent={user_agent!r}\n"
    with open(CONSENT_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def _save_pcm_as_wav(pcm_bytes: bytes, path: str) -> None:
    if not pcm_bytes:
        return
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def _have_voices() -> bool:
    return any(Path(VOICES_DIR).glob("*.mp3"))


def _reset_workspace_outputs() -> None:
    for p in (AUDIO_PATH, TRANSCRIPT_PATH, TIMINGS_PATH):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------- #
#  WebSocket: live recording                                              #
# ---------------------------------------------------------------------- #

@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()

    try:
        hello_raw = await asyncio.wait_for(ws.receive_text(), timeout=15.0)
    except asyncio.TimeoutError:
        await ws.close(code=4000, reason="No hello message")
        return

    try:
        hello = json.loads(hello_raw)
    except json.JSONDecodeError:
        await ws.close(code=4000, reason="Hello must be JSON")
        return

    if not hello.get("consent"):
        await ws.close(code=4001, reason="Consent required")
        return

    if not _have_voices():
        await ws.send_json({"type": "error", "message": "No voices enrolled"})
        await ws.close(code=4003, reason="No voices")
        return

    session_id = uuid.uuid4().hex[:12]
    op_tag = f"live:{session_id}"

    if not _acquire_op(op_tag):
        await ws.send_json({"type": "error", "message": "Another operation is active"})
        await ws.close(code=4002, reason="Busy")
        return

    client_host = ws.client.host if ws.client else "unknown"
    user_agent = hello.get("user_agent", "")
    _log_consent(op_tag, user_agent, client_host)
    log.info("Live %s: started from %s", session_id, client_host)

    _reset_workspace_outputs()
    _notify_speaker_reload()

    audio_source = WebSocketAudioSource()
    result_q: queue.Queue = queue.Queue()

    def on_line(payload: dict) -> None:
        result_q.put(payload)

    pipeline = StreamingPipeline(
        source=audio_source,
        redis_client=_redis(),
        output_path=TRANSCRIPT_PATH,
        timings_path=TIMINGS_PATH,
        streaming_vad=True,
        on_line=on_line,
    )

    await ws.send_json({"type": "session", "session_id": session_id})

    loop = asyncio.get_running_loop()
    pipeline_fut = loop.run_in_executor(None, pipeline.run)

    async def forward_results() -> None:
        while True:
            try:
                payload = await asyncio.to_thread(result_q.get, True, 0.3)
            except queue.Empty:
                if pipeline_fut.done() and result_q.empty():
                    return
                continue
            try:
                await ws.send_json({"type": "transcript", **payload})
            except Exception as e:
                log.warning("Live %s: send_json failed: %s", session_id, e)
                return

    forwarder = asyncio.create_task(forward_results())

    try:
        while True:
            msg = await ws.receive()
            t = msg.get("type")
            if t == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                audio_source.push(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("action") == "stop":
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("Live %s: WS error: %s", session_id, e)
    finally:
        audio_source.close()
        try:
            await asyncio.wait_for(pipeline_fut, timeout=SESSION_FINALIZE_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("Live %s: pipeline did not finish within %.0fs — releasing lock anyway",
                        session_id, SESSION_FINALIZE_TIMEOUT_S)
        forwarder.cancel()
        try:
            await forwarder
        except asyncio.CancelledError:
            pass

        try:
            _save_pcm_as_wav(audio_source.recorded_pcm(), AUDIO_PATH)
            log.info("Live %s: audio saved to %s", session_id, AUDIO_PATH)
        except Exception as e:
            log.warning("Live %s: failed to save audio: %s", session_id, e)

        try:
            await ws.send_json({"type": "session_end", "session_id": session_id})
            await ws.close()
        except Exception:
            pass

        _release_op(op_tag)
        log.info("Live %s: closed", session_id)


# ---------------------------------------------------------------------- #
#  Upload mp3: batched processing with progress over WS                   #
# ---------------------------------------------------------------------- #

# Global single-slot progress queue (one upload at a time, gated by op lock).
_upload_progress: queue.Queue[dict] = queue.Queue()
_upload_active = threading.Event()
_upload_error: dict = {}  # {"message": str} when latest upload errored


def _run_upload_pipeline(mp3_bytes: bytes, op_tag: str) -> None:
    tmp_mp3 = os.path.join(WORKSPACE_DIR, "_upload_tmp.mp3")
    try:
        # Persist input + convert to wav for the workspace audio.
        with open(tmp_mp3, "wb") as f:
            f.write(mp3_bytes)

        audio = AudioSegment.from_file(tmp_mp3)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(SAMPLE_WIDTH)
        audio.export(AUDIO_PATH, format="wav")
        log.info("Upload: input mp3 converted to %s (%d ms)", AUDIO_PATH, len(audio))

        def on_line(payload: dict) -> None:
            _upload_progress.put({"type": "transcript", **payload})

        pipeline = StreamingPipeline(
            source=FileAudioSource(tmp_mp3, realtime_factor=0.0),
            redis_client=_redis(),
            output_path=TRANSCRIPT_PATH,
            timings_path=TIMINGS_PATH,
            streaming_vad=False,  # batched VAD for offline-style processing
            on_line=on_line,
        )
        pipeline.run()
        _upload_progress.put({"type": "complete"})
    except Exception as e:
        log.exception("Upload pipeline failed: %s", e)
        _upload_error["message"] = str(e)
        _upload_progress.put({"type": "error", "message": str(e)})
    finally:
        try:
            os.remove(tmp_mp3)
        except FileNotFoundError:
            pass
        _upload_active.clear()
        _release_op(op_tag)


@app.post("/upload_audio")
async def upload_audio(file: UploadFile, consent: str = Form("")) -> dict:
    if consent.lower() not in ("true", "1", "yes"):
        raise HTTPException(status_code=400, detail="Consent required")
    if not _have_voices():
        raise HTTPException(status_code=400, detail="No voices enrolled")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    task_id = uuid.uuid4().hex[:12]
    op_tag = f"upload:{task_id}"
    if not _acquire_op(op_tag):
        raise HTTPException(status_code=409, detail="Another operation is active")

    _reset_workspace_outputs()
    _notify_speaker_reload()

    # Drain any stale progress events.
    while not _upload_progress.empty():
        try:
            _upload_progress.get_nowait()
        except queue.Empty:
            break
    _upload_error.clear()
    _upload_active.set()

    _log_consent(op_tag, "upload", "n/a")
    log.info("Upload %s: started (%d bytes)", task_id, len(data))

    threading.Thread(
        target=_run_upload_pipeline, args=(data, op_tag), daemon=True, name=f"upload-{task_id}"
    ).start()

    return {"task_id": task_id, "status": "processing"}


@app.websocket("/progress")
async def progress(ws: WebSocket) -> None:
    """Listen-only stream of upload-pipeline progress events."""
    await ws.accept()
    try:
        while True:
            if not _upload_active.is_set() and _upload_progress.empty():
                await ws.send_json({"type": "idle"})
                break
            try:
                payload = await asyncio.to_thread(_upload_progress.get, True, 0.3)
            except queue.Empty:
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                return
            if payload.get("type") in ("complete", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------- #
#  Downloads                                                              #
# ---------------------------------------------------------------------- #

@app.get("/transcript")
def download_transcript() -> FileResponse:
    if not os.path.exists(TRANSCRIPT_PATH):
        raise HTTPException(status_code=404, detail="No transcript")
    return FileResponse(
        TRANSCRIPT_PATH,
        media_type="text/plain; charset=utf-8",
        filename="transcript.txt",
    )


@app.get("/audio")
def download_audio() -> FileResponse:
    if not os.path.exists(AUDIO_PATH):
        raise HTTPException(status_code=404, detail="No audio")
    return FileResponse(AUDIO_PATH, media_type="audio/wav", filename="audio.wav")


@app.get("/state")
def workspace_state() -> dict:
    return {
        "has_transcript": os.path.exists(TRANSCRIPT_PATH),
        "has_audio": os.path.exists(AUDIO_PATH),
        "active_op": _active_op,
        "upload_in_progress": _upload_active.is_set(),
    }
