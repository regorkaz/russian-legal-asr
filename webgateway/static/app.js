// Legal-ASR web client. Single workspace, two modes (live / upload).

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- //
//  Consent gate                                                     //
// ---------------------------------------------------------------- //

const consent = $("consent-checkbox");
const workspace = $("workspace");
consent.addEventListener("change", () => {
    workspace.classList.toggle("disabled", !consent.checked);
    refreshStartButtons();
});

// ---------------------------------------------------------------- //
//  Mode tabs (live vs upload)                                       //
// ---------------------------------------------------------------- //

document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".mode-tab").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".mode-pane").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        $(`mode-${btn.dataset.mode}`).classList.add("active");
    });
});

// ---------------------------------------------------------------- //
//  Audio capture helpers                                            //
// ---------------------------------------------------------------- //

async function openMicStream() {
    return await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
}

async function createPcmPipeline(stream, onFrame) {
    const audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule("/static/worklet.js");

    const src = audioCtx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(audioCtx, "pcm-downsampler");
    node.port.onmessage = (e) => onFrame(e.data);

    src.connect(node);
    node.connect(audioCtx.destination);

    return {
        stop: async () => {
            try { node.disconnect(); } catch (_) {}
            try { src.disconnect(); } catch (_) {}
            stream.getTracks().forEach((t) => t.stop());
            await audioCtx.close();
        },
    };
}

function int16BufferToWav(pcmBuffers, sampleRate = 16000) {
    const totalSamples = pcmBuffers.reduce((a, b) => a + b.byteLength / 2, 0);
    const totalBytes = totalSamples * 2;
    const buffer = new ArrayBuffer(44 + totalBytes);
    const view = new DataView(buffer);
    const writeStr = (offset, s) => {
        for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + totalBytes, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, totalBytes, true);
    let offset = 44;
    for (const buf of pcmBuffers) {
        new Uint8Array(buffer, offset, buf.byteLength).set(new Uint8Array(buf));
        offset += buf.byteLength;
    }
    return new Blob([buffer], { type: "audio/wav" });
}

// ---------------------------------------------------------------- //
//  Voices                                                           //
// ---------------------------------------------------------------- //

const voiceNameInput = $("voice-name");
const voiceRecordBtn = $("voice-record-btn");
const voiceRecordStatus = $("voice-record-status");
const voicePreview = $("voice-preview");
const voiceRecordedActions = $("voice-recorded-actions");
const voiceSaveRecordedBtn = $("voice-save-recorded-btn");
const voiceDiscardBtn = $("voice-discard-btn");
const voiceFileInput = $("voice-file-input");
const voiceUploadBtn = $("voice-upload-btn");
const voiceUploadStatus = $("voice-upload-status");

let voiceRecording = null;   // { pipelineStop, buffers }
let voiceBlob = null;
let voicesCount = 0;

async function refreshVoicesList() {
    try {
        const resp = await fetch("/voices");
        const data = await resp.json();
        const list = $("voices-list");
        list.innerHTML = "";
        voicesCount = data.voices.length;
        if (voicesCount === 0) {
            list.innerHTML = '<li class="muted">Пока нет слепков. Добавьте минимум один.</li>';
        } else {
            data.voices.forEach((v) => {
                const li = document.createElement("li");
                const name = document.createElement("span");
                name.innerHTML = `<b>${v.label}</b> <span class="muted small">(${(v.duration_ms/1000).toFixed(1)} с)</span>`;
                li.appendChild(name);
                const del = document.createElement("button");
                del.className = "ghost";
                del.textContent = "Удалить";
                del.addEventListener("click", async () => {
                    await fetch(`/voices/${encodeURIComponent(v.label)}`, { method: "DELETE" });
                    refreshVoicesList();
                });
                li.appendChild(del);
                list.appendChild(li);
            });
        }
        refreshStartButtons();
    } catch (e) {
        console.error(e);
    }
}

function getVoiceName() {
    const name = voiceNameInput.value.trim();
    if (!name) {
        voiceRecordStatus.textContent = "Сначала укажите имя";
        voiceRecordStatus.className = "status err";
        return null;
    }
    return name;
}

async function uploadVoiceBlob(name, blob, statusEl) {
    const fd = new FormData();
    fd.append("file", blob, `${name}.wav`);
    statusEl.textContent = "Сохраняю…";
    statusEl.className = "status";
    try {
        const resp = await fetch(`/voices/${encodeURIComponent(name)}`, {
            method: "PUT",
            body: fd,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || resp.statusText);
        statusEl.textContent = `Сохранено: ${data.label}`;
        statusEl.className = "status ok";
        voiceNameInput.value = "";
        refreshVoicesList();
        return true;
    } catch (e) {
        statusEl.textContent = `Ошибка: ${e.message}`;
        statusEl.className = "status err";
        return false;
    }
}

voiceRecordBtn.addEventListener("click", async () => {
    if (voiceRecording) {
        await voiceRecording.pipelineStop();
        voiceBlob = int16BufferToWav(voiceRecording.buffers);
        voicePreview.src = URL.createObjectURL(voiceBlob);
        voicePreview.style.display = "block";
        voiceRecordedActions.style.display = "flex";
        voiceRecordBtn.textContent = "● Записать с микрофона (5–10 сек)";
        voiceRecordStatus.textContent = `Готово (${(voiceBlob.size/1024).toFixed(1)} KB)`;
        voiceRecordStatus.className = "status";
        voiceRecording = null;
        return;
    }
    if (!getVoiceName()) return;
    try {
        const stream = await openMicStream();
        const buffers = [];
        const pipeline = await createPcmPipeline(stream, (buf) => buffers.push(buf));
        voiceRecording = { pipelineStop: pipeline.stop, buffers };
        voiceRecordBtn.textContent = "■ Остановить запись";
        voiceRecordStatus.textContent = "Запись…";
        voiceRecordStatus.className = "status live";
    } catch (e) {
        voiceRecordStatus.textContent = `Ошибка микрофона: ${e.message}`;
        voiceRecordStatus.className = "status err";
    }
});

voiceDiscardBtn.addEventListener("click", () => {
    voiceBlob = null;
    voicePreview.style.display = "none";
    voicePreview.src = "";
    voiceRecordedActions.style.display = "none";
    voiceRecordStatus.textContent = "";
});

voiceSaveRecordedBtn.addEventListener("click", async () => {
    const name = getVoiceName();
    if (!name || !voiceBlob) return;
    const ok = await uploadVoiceBlob(name, voiceBlob, voiceRecordStatus);
    if (ok) voiceDiscardBtn.click();
});

voiceUploadBtn.addEventListener("click", () => voiceFileInput.click());
voiceFileInput.addEventListener("change", async () => {
    const f = voiceFileInput.files?.[0];
    if (!f) return;
    const name = getVoiceName();
    if (!name) { voiceFileInput.value = ""; return; }
    await uploadVoiceBlob(name, f, voiceUploadStatus);
    voiceFileInput.value = "";
});

refreshVoicesList();

// ---------------------------------------------------------------- //
//  Transcript rendering                                             //
// ---------------------------------------------------------------- //

const transcriptEl = $("transcript");
const downloadsEl = $("downloads");

function clearTranscript() {
    transcriptEl.innerHTML = "";
    downloadsEl.style.display = "none";
}

function appendBubble(payload) {
    const li = document.createElement("div");
    li.className = "bubble";
    li.dataset.speaker = payload.speaker;
    if (payload.speaker !== "Unknown") li.classList.add("speaker-known");

    const meta = document.createElement("div");
    meta.className = "meta";
    const range = (payload.start_s != null && payload.end_s != null)
        ? `${payload.start_s.toFixed(1)}s – ${payload.end_s.toFixed(1)}s`
        : "";
    const conf = payload.speaker_confidence != null
        ? ` · conf ${Number(payload.speaker_confidence).toFixed(2)}`
        : "";
    meta.textContent = `[${payload.speaker}] ${range}${conf}`;

    const text = document.createElement("div");
    text.className = "text";
    text.textContent = payload.text || "(пусто)";

    li.appendChild(meta);
    li.appendChild(text);
    transcriptEl.appendChild(li);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function showDownloads() {
    // Force a fresh fetch when clicking, so caches don't return stale files.
    const ts = Date.now();
    $("download-transcript").href = `/transcript?_=${ts}`;
    $("download-audio").href = `/audio?_=${ts}`;
    downloadsEl.style.display = "flex";
}

// ---------------------------------------------------------------- //
//  Live mode                                                        //
// ---------------------------------------------------------------- //

const liveStartBtn = $("live-start-btn");
const liveStopBtn = $("live-stop-btn");
const liveStatus = $("live-status");

let liveSession = null;

liveStartBtn.addEventListener("click", async () => {
    if (liveSession) return;
    if (voicesCount === 0) {
        liveStatus.textContent = "Сначала добавьте хотя бы один слепок";
        liveStatus.className = "status err";
        return;
    }
    clearTranscript();

    try {
        const stream = await openMicStream();
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        const ws = new WebSocket(`${proto}//${location.host}/stream`);
        ws.binaryType = "arraybuffer";

        ws.addEventListener("open", () => {
            ws.send(JSON.stringify({ consent: true, user_agent: navigator.userAgent }));
        });

        let pipelineStop = null;
        ws.addEventListener("message", (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                if (msg.type === "session") {
                    liveStatus.textContent = "Запись…";
                    liveStatus.className = "status live";
                } else if (msg.type === "transcript") {
                    appendBubble(msg);
                } else if (msg.type === "error") {
                    liveStatus.textContent = `Ошибка: ${msg.message}`;
                    liveStatus.className = "status err";
                } else if (msg.type === "session_end") {
                    liveStatus.textContent = "Завершена";
                    liveStatus.className = "status ok";
                    showDownloads();
                }
            } catch (e) { console.error("Bad WS message", e); }
        });

        ws.addEventListener("close", async () => {
            if (pipelineStop) { try { await pipelineStop(); } catch (_) {} pipelineStop = null; }
            liveStartBtn.disabled = false;
            liveStopBtn.disabled = true;
            liveSession = null;
            if (!liveStatus.classList.contains("ok") && !liveStatus.classList.contains("err")) {
                liveStatus.textContent = "Соединение закрыто";
                liveStatus.className = "status";
                showDownloads();
            }
        });

        ws.addEventListener("error", () => {
            liveStatus.textContent = "Ошибка WebSocket";
            liveStatus.className = "status err";
        });

        await new Promise((resolve, reject) => {
            if (ws.readyState === WebSocket.OPEN) return resolve();
            ws.addEventListener("open", resolve, { once: true });
            ws.addEventListener("error", reject, { once: true });
        });

        const pipeline = await createPcmPipeline(stream, (buf) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(buf);
        });
        pipelineStop = pipeline.stop;
        liveSession = { ws, pipelineStop };
        liveStartBtn.disabled = true;
        liveStopBtn.disabled = false;
    } catch (e) {
        liveStatus.textContent = `Ошибка: ${e.message}`;
        liveStatus.className = "status err";
    }
});

liveStopBtn.addEventListener("click", async () => {
    if (!liveSession) return;
    liveStopBtn.disabled = true;
    liveStatus.textContent = "Останавливаю…";
    liveStatus.className = "status";
    try { await liveSession.pipelineStop(); } catch (_) {}
    liveSession.pipelineStop = null;
    try {
        if (liveSession.ws.readyState === WebSocket.OPEN) {
            liveSession.ws.send(JSON.stringify({ action: "stop" }));
        }
    } catch (_) {}
    setTimeout(() => {
        if (liveSession && liveSession.ws.readyState !== WebSocket.CLOSED) {
            try { liveSession.ws.close(); } catch (_) {}
        }
    }, 45000);
});

// ---------------------------------------------------------------- //
//  Upload mode                                                      //
// ---------------------------------------------------------------- //

const uploadPickBtn = $("upload-pick-btn");
const uploadFileInput = $("upload-file-input");
const uploadFilename = $("upload-filename");
const uploadStartBtn = $("upload-start-btn");
const uploadStatus = $("upload-status");

let uploadFile = null;

uploadPickBtn.addEventListener("click", () => uploadFileInput.click());
uploadFileInput.addEventListener("change", () => {
    uploadFile = uploadFileInput.files?.[0] || null;
    uploadFilename.textContent = uploadFile ? uploadFile.name : "";
    uploadStartBtn.disabled = !uploadFile;
});

uploadStartBtn.addEventListener("click", async () => {
    if (!uploadFile) return;
    if (voicesCount === 0) {
        uploadStatus.textContent = "Сначала добавьте хотя бы один слепок";
        uploadStatus.className = "status err";
        return;
    }
    clearTranscript();
    uploadStartBtn.disabled = true;
    uploadStatus.textContent = "Загружаю файл…";
    uploadStatus.className = "status";

    // Open WS BEFORE POST so we don't miss early progress.
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/progress`);
    ws.addEventListener("message", (ev) => {
        try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "transcript") {
                appendBubble(msg);
            } else if (msg.type === "complete") {
                uploadStatus.textContent = "Готово";
                uploadStatus.className = "status ok";
                showDownloads();
                ws.close();
            } else if (msg.type === "error") {
                uploadStatus.textContent = `Ошибка: ${msg.message}`;
                uploadStatus.className = "status err";
                ws.close();
            } else if (msg.type === "idle") {
                // No active upload yet; close — POST will replace this WS by reopening
            }
        } catch (e) { console.error(e); }
    });
    ws.addEventListener("close", () => {
        uploadStartBtn.disabled = false;
    });

    await new Promise((resolve) => {
        if (ws.readyState === WebSocket.OPEN) return resolve();
        ws.addEventListener("open", resolve, { once: true });
    });

    const fd = new FormData();
    fd.append("file", uploadFile);
    fd.append("consent", "true");
    try {
        const resp = await fetch("/upload_audio", { method: "POST", body: fd });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || resp.statusText);
        uploadStatus.textContent = "Обрабатываю (RTF=0)…";
        uploadStatus.className = "status live";
    } catch (e) {
        uploadStatus.textContent = `Ошибка: ${e.message}`;
        uploadStatus.className = "status err";
        try { ws.close(); } catch (_) {}
        uploadStartBtn.disabled = false;
    }
});

// ---------------------------------------------------------------- //
//  Start-button gating                                              //
// ---------------------------------------------------------------- //

function refreshStartButtons() {
    const ready = consent.checked && voicesCount > 0;
    liveStartBtn.disabled = !ready || liveSession !== null;
    uploadStartBtn.disabled = !ready || !uploadFile;
}
