# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1 - compile llama.cpp with CUDA support
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv \
    git wget curl \
    build-essential cmake g++ ninja-build \
    libcurl4-openssl-dev pkg-config ccache zstd \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

ARG LLAMA_CPP_TAG=b8863
RUN git clone https://github.com/ggml-org/llama.cpp.git /llama.cpp \
    && cd /llama.cpp \
    && git checkout tags/${LLAMA_CPP_TAG} -b llmmll

WORKDIR /llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j6

# ---------------------------------------------------------------------------
# Stage 2 - runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Pull the uv binary from the official image (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    SHARED_VENV="/opt/venv/shared" \
    UV_PROJECT_ENVIRONMENT="/opt/venv/shared" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/cuda/lib64" \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=0 \
    RUNNER_PORT=8000 \
    RUNNER_HOST=0.0.0.0

# Runtime-only apt deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    curl libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Compiled llama.cpp binaries
COPY --from=llama-builder /llama.cpp /llama.cpp

WORKDIR /app

# Create the shared venv uv will manage
RUN uv venv --python 3.12 ${SHARED_VENV}

# Copy only the dep manifests first
COPY pyproject.toml ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev || true

# Copy application source
COPY . ./

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
