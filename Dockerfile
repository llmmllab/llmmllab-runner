# syntax=docker/dockerfile:1.7
#
# llmmllab-runner Dockerfile.
#
# Builds the two native inference servers used by the runner — llama.cpp and
# stable-diffusion.cpp — from their pinned git submodules under vendors/, then
# packages them with the Python service into a CUDA 12.8 runtime image.
#
# Submodules
# ----------
#   vendors/llama.cpp              ggml-org/llama.cpp  (text inference)
#   vendors/stable-diffusion.cpp   leejet/stable-diffusion.cpp (image inference)
#
# Both must be initialised before the build (the build context needs the source
# tree on disk). The Makefile target `vendor-sync` runs the right git command
# for you; CI runs it as the first step of `make docker-build`.
#
# To update either vendor, cd into its submodule, check out the desired tag /
# commit, and commit the bumped pointer in the parent repo — there is no tag
# arg to override.

# ---------------------------------------------------------------------------
# Stage 1 — compile llama.cpp with CUDA support from vendors/llama.cpp
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

# Copy the pinned submodule source instead of cloning at build time. Using the
# submodule means the version that gets baked into the image always matches
# what was committed to git — no more "did CI grab a fresh tag?" surprises.
COPY vendors/llama.cpp /llama.cpp

WORKDIR /llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j6

# ---------------------------------------------------------------------------
# Stage 2 — compile stable-diffusion.cpp with CUDA support
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS sd-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl \
    build-essential cmake g++ ninja-build \
    libcurl4-openssl-dev pkg-config ccache zstd \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Pinned by submodule. SD_CUDA target = SM 75 (T4) + SM 86 (RTX A6000 / 3090).
COPY vendors/stable-diffusion.cpp /stable-diffusion.cpp

WORKDIR /stable-diffusion.cpp
RUN cmake -B build \
    -DSD_CUDA=ON \
    -DSD_BUILD_SERVER=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j6

# ---------------------------------------------------------------------------
# Stage 3 — Hunyuan3D-2.1 build (image-to-3D pipeline)
#
# Hunyuan3D ships as a git repo, not a wheel.  Two custom CUDA extensions
# (custom_rasterizer, differentiable_renderer) compile against torch's
# CUDA headers, so this stage needs both the dev toolchain AND torch
# installed in the venv we'll later copy into the runtime image.
#
# We install everything into the same ``/opt/venv/shared`` path the
# runtime stage expects, so the COPY in stage 4 picks it up verbatim.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS hunyuan-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip \
    git wget curl \
    build-essential cmake g++ ninja-build \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs \
    TORCH_CUDA_ARCH_LIST="7.5;8.6" \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Pinned to a specific Hunyuan3D-2 commit — the repo doesn't tag releases.
ARG HUNYUAN3D_REF=main
RUN git clone --depth=1 --branch=${HUNYUAN3D_REF} \
    https://github.com/Tencent/Hunyuan3D-2.git /opt/hunyuan3d

# Use pip directly (Hunyuan3D's setup is pip-flavoured; uv would re-resolve
# every transitive dep and we'd lose the requirements.txt pin set).
# Skip ``pip install --upgrade pip`` — Ubuntu's debian-managed pip refuses
# self-uninstall (no RECORD file) and the 24.0 system pip is new enough.
RUN python -m pip install --no-cache-dir --upgrade setuptools wheel

# torch must be installed BEFORE the custom-rasterizer setup.py runs —
# its ``setup.py`` imports torch at build time to discover include paths.
# Pin to the CUDA 12.1 wheel (closest pre-built variant to our 12.8
# runtime — CUDA minor versions are forward-compatible).
RUN python -m pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

WORKDIR /opt/hunyuan3d
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir -e .

# Custom CUDA extensions — build against torch's headers.
RUN cd hy3dgen/texgen/custom_rasterizer && python setup.py install
RUN cd hy3dgen/texgen/differentiable_renderer && python setup.py install || \
    echo "WARN: differentiable_renderer build failed; texture gen disabled"

# ---------------------------------------------------------------------------
# Stage 4 — runtime image
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
    RUNNER_HOST=0.0.0.0 \
    LLAMA_SERVER_EXECUTABLE=/llama.cpp/build/bin/llama-server \
    SD_SERVER_EXECUTABLE=/stable-diffusion.cpp/build/bin/sd-server \
    HUNYUAN3D_MODEL_PATH=/models/hunyuan3d \
    SD_OUTPUT_DIR=/tmp/sd-out

# Runtime-only apt deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    curl libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Compiled binaries from the build stages. We copy the entire build tree so
# any shared libraries (libggml*.so, libstable-diffusion.so) stay alongside
# their executables — sd-server in particular links against several .so files
# emitted next to it in build/bin/.
COPY --from=llama-builder /llama.cpp /llama.cpp
COPY --from=sd-builder /stable-diffusion.cpp /stable-diffusion.cpp

# Hunyuan3D source + the system-site-packages where it was installed
# (pip install -e .).  We also pull in the system python's dist-packages
# from the hunyuan-builder stage so the compiled custom_rasterizer /
# differentiable_renderer extensions are available without a re-install.
COPY --from=hunyuan-builder /opt/hunyuan3d /opt/hunyuan3d
COPY --from=hunyuan-builder /usr/local/lib/python3.12/dist-packages \
    /usr/local/lib/python3.12/dist-packages
# torch ships ~3 GB of CUDA libs that the runtime image's cudnn-runtime
# base already has, but the wheels reference them via their own bundled
# copies.  We keep both — disk is cheap relative to compilation time.

WORKDIR /app

# Create the shared venv uv will manage.  ``--system-site-packages``
# is critical: it lets the venv see ``hy3dgen`` + ``torch`` that we
# installed into ``/usr/local/lib/python3.12/dist-packages`` via the
# hunyuan-builder stage above, without needing to re-install
# 3 GB of CUDA wheels through ``uv pip install``.
RUN uv venv --python 3.12 --system-site-packages ${SHARED_VENV}

# Copy only the dep manifests first
COPY pyproject.toml ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev || true

# Copy application source. We strip vendors/ from the runtime image — the
# native binaries live under /llama.cpp and /stable-diffusion.cpp (copied
# from the build stages above) and the C++ source trees would otherwise
# add ~2 GB of unused artefacts to every image layer.
COPY . ./
RUN rm -rf /app/vendors

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
