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
#
# Cache-bust keyed on the llama.cpp submodule commit. Docker was serving the
# COPY+compile layers from cache even after the submodule was bumped (the COPY
# cache key didn't reflect the new content on the self-hosted builder), so a
# bumped llama.cpp never actually recompiled. Passing --build-arg LLAMA_SHA=<sha>
# (= `git -C vendors/llama.cpp rev-parse HEAD`) invalidates these layers ONLY
# when the submodule SHA changes, forcing a recompile then (and only then).
ARG LLAMA_SHA=unknown
RUN echo "Building llama.cpp at submodule ${LLAMA_SHA}"
COPY vendors/llama.cpp /llama.cpp

WORKDIR /llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NCCL=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build --config Release -t llama-server -j6

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
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target sd-server -j6

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

# Hunyuan3D-2 is vendored as a git submodule at ``vendors/Hunyuan3D-2``
# (pinned SHA in .gitmodules + the parent repo's index).  Copy it into
# the builder so its setup.py + CUDA extensions can build, and so the
# editable install in ``/usr/local/lib/python3.12/dist-packages`` keeps
# pointing at the right path when stage 4 copies dist-packages over.
COPY vendors/Hunyuan3D-2 /opt/hunyuan3d

# Use pip directly (Hunyuan3D's setup is pip-flavoured; uv would re-resolve
# every transitive dep and we'd lose the requirements.txt pin set).
# We skip ``pip install --upgrade pip setuptools wheel`` — Ubuntu's
# debian-managed packages refuse self-uninstall (no RECORD file) and the
# shipped versions (pip 24.0, setuptools 68.x, wheel 0.42) are new
# enough for everything Hunyuan3D needs.

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

# briaai/RMBG-2.0 (BiRefNet) deps — its trust_remote_code modeling file
# imports kornia + timm.  Tiny relative to torch so no separate stage.
RUN python -m pip install --no-cache-dir kornia timm

# Hunyuan3D-Part (XPart + P3-SAM) deps — mesh decomposition pipeline.
# Adds ~600 MB on top of Hunyuan3D-2.1's environment.
#
# - spconv-cu124    sparse 3D convolutions used by P3-SAM and XPart's
#                   point-cloud encoder.  cu124 wheel is forward-compatible
#                   with our CUDA 12.8 runtime (CUDA minor versions are
#                   ABI-stable for compute libraries).
# - fpsample        farthest-point sampling for surface point sets
# - addict, easydict   YAML-driven config containers XPart uses to
#                       instantiate sub-modules
# - torch_cluster, torch_scatter   pulled from the PyG wheel index, must
#                                   match torch 2.5 + CUDA 12.1 (same as
#                                   the torch wheel above)
RUN python -m pip install --no-cache-dir \
    spconv-cu124 fpsample addict easydict scikit-learn

# ``plyfile`` is trimesh's optional PLY backend.  Without it,
# XPart's per-part marching-cubes export
# (``trimesh.load(buf, format='ply')``) crashes with
# ``Unknown format for load: ply`` and every diffusion-produced part
# silently fails to materialise — the obj_mesh comes back as an
# empty Scene even though P3-SAM segmentation + DiT diffusion both
# succeeded.  Hunyuan3D-2's requirements.txt installs trimesh but
# not plyfile, so add it explicitly.
RUN python -m pip install --no-cache-dir plyfile
RUN python -m pip install --no-cache-dir \
    torch_cluster torch_scatter \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html || \
    echo "WARN: torch_cluster/torch_scatter PyG wheels not available; falling back to source build (slow)"

# flash-attn — required by XPart's Sonata point-transformer encoder.
# Building from source needs nvcc + 30+ min of compile time; instead
# install Dao-AILab's prebuilt wheel matching torch 2.5 + CUDA 12 +
# CPython 3.12 (the combo this image already uses).  ABI flag must be
# ``cxx11abiFALSE`` to match the torch wheel we install above.
# pip rejects ``flash_attn.whl`` (filename must encode the version),
# so save under the canonical wheel name.  ``+cu12torch2.5cxx11abiFALSE``
# is a local-version tag in PEP 440 terms; pip accepts it in the local
# install path without resolving against PyPI.
RUN curl -sSL -o /tmp/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    python -m pip install --no-cache-dir --no-deps /tmp/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl && \
    rm /tmp/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl

# Hunyuan3D-Part source tree.  Neither XPart nor P3-SAM ships a
# setup.py at their root — only chamfer3D under
# P3-SAM/utils/chamfer3D has one.  We:
#   1. COPY the whole vendored tree into /opt/hunyuan3d-part
#   2. Patch P3-SAM/model.py to use explicit ``partgen.models`` /
#      ``partgen.utils.misc`` import paths (the upstream code does
#      ``from models import sonata`` which would collide with our
#      app's own ``models`` package on PYTHONPATH).
#   3. Build chamfer3D as a CUDA extension (P3-SAM's hard dep)
#   4. Add XPart/ and P3-SAM/ to PYTHONPATH at runtime so
#      ``from partgen.partformer_pipeline import PartFormerPipeline``
#      and ``from utils.chamfer3D import ...`` resolve.
#
# This is the same pattern Hunyuan3D-2 itself uses for its custom
# rasterizer + renderer extensions — directory-on-PATH, not wheel.
COPY vendors/Hunyuan3D-Part /opt/hunyuan3d-part
# Four sed patches to upstream P3-SAM / XPart code:
#
# 1. ``from models import sonata`` → ``from partgen.models import sonata``
#    (avoids collision with /app/models/, our app's pydantic package).
# 2. ``from utils.misc import ...`` → ``from partgen.utils.misc import ...``
#    (same collision with /app/utils/).
# 3. ``sonata.load("sonata", repo_id="facebook/sonata", ...)`` →
#    ``sonata.load("/models/sonata/sonata.pth", ...)`` so the model
#    loads from the pre-downloaded weight at /models/sonata/sonata.pth
#    instead of pulling from HuggingFace on first request.  This keeps
#    the runner offline-capable for the part pipeline.
# 4. ``self.model_parallel = torch.nn.DataParallel(self.model)`` →
#    ``self.model_parallel = self.model`` in the bbox predictor.
#    Upstream wraps its YSAM segmenter in DataParallel for multi-GPU
#    batch splitting, but DataParallel hardcodes cuda:0 as the
#    primary device and expects all params there.  We do our own
#    cross-GPU placement (see ``Hunyuan3DPartPipeline._load`` —
#    AlignDevicesHook on the diffusion DiT), so DataParallel here
#    just gets in the way and crashes with "module must have its
#    parameters and buffers on device cuda:0 but found one on cuda:N".
#    Aliasing to the raw model bypasses the wrapper without
#    changing the call sites that reference ``model_parallel``.
RUN sed -i \
        -e 's|^from models import sonata|from partgen.models import sonata|' \
        -e 's|^from utils\.misc import smart_load_model|from partgen.utils.misc import smart_load_model|' \
        -e 's|sonata\.load("sonata", repo_id="facebook/sonata"[^)]*)|sonata.load("/models/sonata/sonata.pth")|' \
        /opt/hunyuan3d-part/P3-SAM/model.py && \
    sed -i 's|self\.model_parallel = torch\.nn\.DataParallel(self\.model)|self.model_parallel = self.model|' \
        /opt/hunyuan3d-part/XPart/partgen/bbox_estimator/auto_mask_api.py \
        /opt/hunyuan3d-part/P3-SAM/demo/auto_mask.py \
        /opt/hunyuan3d-part/P3-SAM/demo/auto_mask_no_postprocess.py && \
    # NOTE: We do NOT change ``num_points=81920`` in
    # ``check_inputs``.  That dictates the surface sample count the
    # XPart shape VAE encoder consumes — it's hardcoded into the
    # autoencoder's ``torch.split(pc, [pc_size, pc_sharpedge_size])``
    # which expects pc.shape[1] == 81920.  Reducing it crashes with
    # ``split_with_sizes expects split_sizes to sum exactly to 32768
    # (input tensor's size at dimension 1), but got split_sizes=
    # [81920, 0]``.  Keep the upstream value; the fp16 cast on the
    # conditioner does the memory savings instead.
    # P3-SAM's bbox predictor batches prompts at ``bs=64`` inside
    # mesh_sam.  At each iteration the segmenter forward holds
    # *multiple* ``[N=100000, K, *]`` intermediate tensors
    # simultaneously — feats_seg (518-wide), feats_seg_2 (521-wide),
    # feats_seg_3, feats_iou (1033-wide), pred_mask*, plus the
    # repeated [N, K, 512] feats themselves.  Combined working-set
    # at K=64 is >20 GB; at K=16 still ~6 GB; at K=4 ~1.5 GB.  Plus
    # we shrink the mesh point sample (``point_num`` default
    # 100000) since the segmenter's working-set scales linearly in
    # both dims.
    #
    # Patched values:
    #   bs:        64    → 4    (16× loop length, ~16× memory drop)
    #   point_num: 100000 → 50000 (2× memory drop, quality
    #              acceptable for coarse-grained part bboxes)
    #
    # Tune higher when on a card with substantial headroom.
    sed -i \
        -e 's|^        bs = 64$|        bs = 4|' \
        -e 's|point_num=100000|point_num=50000|g' \
        /opt/hunyuan3d-part/XPart/partgen/bbox_estimator/auto_mask_api.py
RUN cd /opt/hunyuan3d-part/P3-SAM/utils/chamfer3D && python setup.py install || \
    echo "WARN: chamfer3D build failed; P3-SAM will surface a clean error on first request"

# ---------------------------------------------------------------------------
# Stage 4 — runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Pull the uv binary from the official image (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/opt/hunyuan3d-part/XPart:/opt/hunyuan3d-part/P3-SAM" \
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
    HY3DGEN_MODELS=/models \
    HUNYUAN3D_MODEL_PATH=hunyuan3d \
    HUNYUAN3D_SUBFOLDER=hunyuan3d-dit-v2-1 \
    HUNYUAN3D_USE_SAFETENSORS=false \
    SD_OUTPUT_DIR=/tmp/sd-out

# Runtime-only apt deps (no compilers).
#
# ``libgl1`` is required at runtime by pymeshlab (pulled in transitively
# via hy3dgen.shapegen.postprocessors) — without it the import fails
# with ``libGL.so.1: cannot open shared object file``.
# ``libglvnd0`` provides ``libOpenGL.so.0`` which pymeshlab's
# ``libio_base.so`` plugin needs (separate from ``libGL.so.1``; the
# former is GLVND vendor-neutral, the latter is the GLX-bound legacy
# path).  Missing libOpenGL.so.0 silently disables pymeshlab's PLY
# I/O backend, which breaks Hunyuan3D-Part: XPart roundtrips
# diffusion outputs through PLY via pymeshlab in
# ``pymeshlab2trimesh``, so every per-part export errors with
# ``Unknown format for load: ply`` and the assembled scene is empty.
# ``libglib2.0-0`` is already there for opencv-style deps; ``libgomp1``
# is the OpenMP runtime ggml uses for CPU paths.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    curl libglib2.0-0 libgomp1 libgl1 libglvnd0 libopengl0 \
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
# Hunyuan3D-Part source tree (XPart/partgen + P3-SAM/utils).  Lives at
# the same path the builder stage compiled chamfer3D against, so the
# editable-import path resolves identically.  PYTHONPATH (set above)
# adds both XPart/ and P3-SAM/ to sys.path at runtime.
COPY --from=hunyuan-builder /opt/hunyuan3d-part /opt/hunyuan3d-part
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

# Copy only the dep manifests first.  uv.lock is committed so
# ``uv sync --frozen`` is reproducible across builds; failure here is
# a real error (not "fall back to system pip") because the runner
# depends on real fastapi/uvicorn/structlog/etc resolved from this
# lockfile, not whatever Hunyuan3D's requirements.txt happened to pin.
COPY pyproject.toml uv.lock ./

# Install dependencies (production set; dev group skipped via --no-dev).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy application source. We strip vendors/ from the runtime image — the
# native binaries live under /llama.cpp and /stable-diffusion.cpp (copied
# from the build stages above) and the C++ source trees would otherwise
# add ~2 GB of unused artefacts to every image layer.
COPY . ./
RUN rm -rf /app/vendors

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "30"]
