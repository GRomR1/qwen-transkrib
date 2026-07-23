# syntax=docker/dockerfile:1.7

# ---- builder ----
FROM python:3.12-slim AS builder
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY packages ./packages
RUN uv sync --frozen --no-dev
COPY src ./src
RUN uv pip install --no-deps .

# ---- runtime: MetaX (torch 2.10 + MACA 3.8) ----
FROM cr.metax-tech.com/public-library/maca-pytorch:3.8.0.11-torch2.10-py312-ubuntu24.04-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/packages /app/packages
COPY --from=builder /app/src /app/src

# The builder venv was created on Debian (python -> /usr/local/bin).
# MetaX image uses conda; patch the venv so it resolves against conda's stdlib.
RUN ln -s /opt/conda/bin/python3 /usr/local/bin/python3 \
    && sed -i 's|^home = .*|home = /opt/conda/bin|' /app/.venv/pyvenv.cfg

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/models \
    HF_TOKEN=""

VOLUME ["/data/in", "/data/out", "/data/models"]
ENTRYPOINT ["qwen-transkrib"]
CMD ["--help"]
