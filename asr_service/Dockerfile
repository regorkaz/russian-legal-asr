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
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pygtrie>=2.1 && \
    pip install --no-cache-dir --no-deps "pyctcdecode>=0.5.0"

# Build KenLM from source (binaries + python bindings) for shallow fusion.
RUN cd /tmp && \
    git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && \
    mkdir build && cd build && \
    cmake .. && make -j4 && \
    cp bin/lmplz bin/build_binary /usr/local/bin/ && \
    cd .. && pip install --no-cache-dir . && \
    cd /tmp && rm -rf kenlm

COPY . .

CMD ["python", "worker.py"]
