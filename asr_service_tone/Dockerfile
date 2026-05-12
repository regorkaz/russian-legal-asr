FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git \
        build-essential cmake libboost-all-dev libeigen3-dev \
        zlib1g-dev libbz2-dev liblzma-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download T-one (t-tech/T-one) so first run doesn't pay HF download.
RUN python -c "from tone import StreamingCTCPipeline; StreamingCTCPipeline.from_hugging_face()"

COPY . .

CMD ["python", "worker.py"]
