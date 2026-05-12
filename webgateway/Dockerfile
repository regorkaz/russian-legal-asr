FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY webgateway/requirements.txt /app/webgateway/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/webgateway/requirements.txt

COPY gateway/ /app/gateway/
COPY webgateway/ /app/webgateway/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "webgateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
