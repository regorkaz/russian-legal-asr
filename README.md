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
- `data/input/consultationN/` — `audio.mp3`, `text.txt` (`[Lawyer]:` / `[Client]:`), `voices/{Lawyer,Client}.mp3`.
- `data/output/<experiment>/` — `transcript.txt`, `metrics.json`, `timings.json`, `benchmark.csv`.
- `data/web_workspace/voices/` — слепки веб-сессий.

## Запуск

Нужен только **Docker** с Docker Compose (модели и зависимости ставятся внутри контейнеров). Команды ниже вводятся последовательно из корня репозитория.

### 0. Подготовка

```bash
git clone <repo-url> russian-legal-asr
cd russian-legal-asr
```

Создать `.env` с токеном HuggingFace — он нужен `speaker_worker` для скачивания `pyannote/embedding` (предварительно примите условия модели на её странице):

```bash
echo "HF_TOKEN=hf_ваш_токен" > .env
```

Разложить данные по консультациям в `data/input/` (модель GigaAM-v3 скачается сама в `data/cache/` при первом старте — около 1 ГБ):

```
data/input/consultation1/
├── audio.mp3            # запись консультации
├── text.txt            # эталон: строки "[Lawyer]: ..." и "[Client]: ..."
└── voices/
    ├── Lawyer.mp3      # образец голоса юриста (для speaker-ID)
    └── Client.mp3      # образец голоса клиента
```

### 1. Веб-интерфейс

```bash
docker compose up -d
```

Открыть **http://localhost:8000** (live-микрофон, загрузка mp3, CRUD слепков голосов). Первый старт дольше — собирается образ и скачиваются модели; статус: `docker compose ps`, логи: `docker compose logs -f`.

Альтернативные ASR-модели (один воркер в эфире):

```bash
docker compose --profile whisper up -d   # Whisper-large-v3
docker compose --profile tone up -d      # T-one (8 кГц)
```

Остановить всё: `docker compose down`.

### 2. Оффлайн-эксперимент конкретной конфигурации

Конфигурация воркеров (ASR-декодинг, режим speaker-ID, пороги) задаётся через `.env` и применяется при `up`. Конфигурация gateway (консультация, VAD) — через переменные перед `docker compose run`.

```bash
# Шаг 1 — задать конфигурацию воркеров в .env (пример: greedy + cosine speaker-ID)
cat >> .env <<'EOF'
LM_MODE=greedy
SPEAKER_MODE=cosine
SIMILARITY_THRESHOLD=0.5
EOF

# Шаг 2 — поднять/пересоздать стек с этой конфигурацией
docker compose up -d

# Шаг 3 — прогнать консультацию через симулятор (одноразовый контейнер)
CONSULTATION=consultation1 EXPERIMENT=exp1 docker compose run --rm simulator
```

Метрики (WER, CER, Speaker accuracy, latency) печатаются в консоль и сохраняются в `data/output/<EXPERIMENT>/` (`transcript.txt`, `metrics.json`, `timings.json`).

Переменные прогона (ставятся перед `docker compose run --rm simulator`):

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `CONSULTATION` | `consultation1` | какую папку из `data/input/` прогнать |
| `EXPERIMENT` | `=CONSULTATION` | имя папки результата в `data/output/` |
| `VAD_MODE` | `batched` | `none` / `batched` / `streaming` |
| `VAD_MIN_SILENCE_MS` | дефолт VAD | пауза для нарезки (напр. `900` для streaming) |
| `REALTIME_FACTOR` | `0.0` | `0.0` — максимально быстро, `1.0` — в реальном времени |

Пример — лучшая конфигурация по итогам ВКР (GigaAM greedy + streaming VAD + online-диаризация):

```bash
# .env: LM_MODE=greedy, SPEAKER_MODE=diarization
docker compose up -d
CONSULTATION=consultation1 EXPERIMENT=best VAD_MODE=streaming VAD_MIN_SILENCE_MS=900 \
  docker compose run --rm simulator
```

> Режимы `LM_MODE=kenlm`/`hotwords` требуют предварительно собранных артефактов (`asr_service/build_kenlm.py`, файл хотвордов) — для базовых прогонов используйте `greedy`.

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
