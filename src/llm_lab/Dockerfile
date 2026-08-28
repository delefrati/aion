FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer — survives source changes)
COPY pyproject.toml /app/llm_lab/pyproject.toml
COPY __init__.py /app/llm_lab/__init__.py
RUN cp /app/llm_lab/pyproject.toml /app/pyproject.toml && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir .

# Copy source (only this layer rebuilds on code changes)
COPY . /app/llm_lab/

CMD ["python", "-m", "llm_lab.cli", "--help"]
