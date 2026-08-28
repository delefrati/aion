FROM python:3.12-slim AS builder

WORKDIR /build

COPY backend/pyproject.toml .
COPY backend/app/ app/
COPY llm_lab/ llm_lab/

RUN pip install --no-cache-dir --prefix=/install \
        "torch==2.13.0" . \
    && find /install -name '*.pyc' -delete \
    && { find /install -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; true; }

# ---- runtime ----
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY backend/app/ app/
COPY llm_lab/ llm_lab/

# HuggingFace cache lives in a volume so weights survive container rebuilds
ENV HF_HOME=/app/hf-cache
# Triton JIT needs to find libcuda.so.1 mounted by nvidia-container-toolkit
ENV LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

EXPOSE 8900
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8900"]
