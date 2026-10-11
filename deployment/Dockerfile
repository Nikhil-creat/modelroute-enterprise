# syntax=docker/dockerfile:1.6
# ModelRoute Enterprise API image. Designed and developed by NIKHIL CHARY SRIRAMOJU.
# Build from the project root:  docker build -f deployment/Dockerfile -t modelroute-api .
# Optional in-image vLLM:       docker build --build-arg WITH_VLLM=1 ...  (compose runs vLLM as its own service)
ARG CUDA_VERSION=12.1.1

# ---------- stage 1: build the virtualenv (PyTorch, HF stack, ChromaDB, FastAPI) ----------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 AS builder
ARG WITH_VLLM=0
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip build-essential git \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip wheel \
    && pip install -r /tmp/requirements.txt \
    && if [ "$WITH_VLLM" = "1" ]; then pip install vllm; fi

# ---------- stage 2: slim CUDA runtime ----------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" HF_HOME=/app/.cache/huggingface MR_DATA_DIR=/app/data MR_ARTIFACT_DIR=/app/artifacts
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 modelroute
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=modelroute:modelroute . /app
RUN mkdir -p /app/data /app/artifacts /app/.cache && chown -R modelroute:modelroute /app
USER modelroute
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
