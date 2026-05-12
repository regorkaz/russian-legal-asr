# Neural Network Module for Legal Consultation Dialogue Recognition

Real-time распознавание русскоязычных юридических консультаций с автоматической разметкой ролей юрист/клиент. Локальный CPU-стек на Docker Compose + Redis с тремя взаимозаменяемыми ASR-моделями (GigaAM-v3, Whisper-large-v3, T-one), Silero VAD, KenLM/hotword postproc и двумя режимами speaker-ID (cosine с регистрацией / online-диаризация). Реализация ВКР.

Лучшая конфигурация по итогам экспериментов: GigaAM-v3 greedy + Silero streaming (`min_silence=900`) + online-диаризация — WER 8.57%, CER 3.55%, Spk acc 89.00%, средняя ASR-latency 488 мс на CPU.

## Архитектура

```
audio (WebSocket / mp3) → gateway (Silero VAD) → Redis
                                                  ├─ tasks:asr     → asr_worker     → results:asr
                                                  └─ tasks:speaker → speaker_worker → results:speaker
                                                                          ↓
                                                                  упорядочивание по seq_num
                                                                          ↓
                                                                      transcript
```

ASR-воркер один в эфире, переключается через docker-профили.

## Структура репозитория

- `gateway/` — стриминговый оркестратор (Silero VAD: `none` / `batched` / `streaming`; producer/sender/collector-потоки) и CLI-симулятор `gateway_simulator.py` для оффлайн-прогонов.
- `asr_service/` — основной воркер GigaAM-v3 (CTC + опциональный pyctcdecode beam search с KenLM и hotwords). `build_kenlm.py` собирает 4-грамм KenLM на корпусе RusLawOD.
- `asr_service_whisper/` — Whisper-large-v3 через faster-whisper, int8. Профиль `whisper`.
- `asr_service_tone/` — T-one (телефонный 8 кГц). Профиль `tone`.
- `speaker_service/` — pyannote/embedding + два режима: `cosine` (по предварительным слепкам) и `diarization` (online speaker bank).
- `webgateway/` — FastAPI + WebSocket: `/` (UI), `/voices` CRUD слепков, `WS /stream` (live-микрофон), `POST /upload_audio` + `WS /progress` (готовый mp3), `/transcript`, `/audio`, `/abort`.
- `metrics/` — WER, CER, Speaker accuracy через `jiwer` с нормализацией.
- `scripts/run_benchmark.py` — прогон по всем `data/input/consultation*`.
- `scripts/sweep.py` — грид-перебор по VAD / cosine threshold / HOTWORD_WEIGHT / KenLM (α, β).
- `data/input/consultationN/` — `audio.mp3`, `text.txt` (`[Lawyer]:` / `[Client]:`), `voices/{Lawyer,Client}.mp3`.
- `data/output/<experiment>/` — `transcript.txt`, `metrics.json`, `timings.json`, `benchmark.csv`.
- `data/web_workspace/voices/` — слепки веб-сессий.

## Запуск

Требуется Docker Compose; `HF_TOKEN` в `.env` для скачивания pyannote/embedding.

```bash
# базовый стек: GigaAM + speaker + web UI (открыть http://localhost:8000)
docker compose up -d

# альтернативные ASR (один воркер в эфире):
docker compose --profile whisper up -d
docker compose --profile tone up -d
```

Оффлайн-симулятор без Docker (Redis должен быть поднят):

```bash
python -m gateway.gateway_simulator        # один файл
python scripts/run_benchmark.py --name 01_baseline   # все консультации
python scripts/sweep.py --mode=vad --name vad_sweep  # vad / cosine / hotword / kenlm
```

## Ключевые env-переменные

| Переменная | Где | Значения / по умолчанию |
|---|---|---|
| `VAD_MODE` | gateway | `none` / `batched` / `streaming` |
| `LM_MODE` | asr_worker (GigaAM) | `greedy` / `hotwords` / `kenlm` / `hotwords_kenlm` |
| `LM_ALPHA`, `LM_BETA`, `LM_BEAM_WIDTH` | asr_worker | `0.5` / `1.5` / `10` |
| `HOTWORD_WEIGHT` | asr_worker | `10.0` |
| `WHISPER_MODEL`, `WHISPER_COMPUTE_TYPE`, `WHISPER_BEAM_SIZE` | asr_worker_whisper | `large-v3` / `int8` / `1` |
| `SPEAKER_MODE` | speaker_worker | `cosine` / `diarization` |
| `SIMILARITY_THRESHOLD` | speaker_worker (cosine) | `0.5` |
| `DIARIZATION_THRESHOLD`, `DIARIZATION_SMOOTHING` | speaker_worker (diarization) | `0.55` / `0.1` |
| `HF_TOKEN` | speaker_worker | токен HuggingFace для pyannote |
