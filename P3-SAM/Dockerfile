# =============================================================================
# P3-SAM API — Docker Image
#
# Target hardware : NVIDIA RTX Pro 6000 (Blackwell, sm_120)
# CUDA toolkit    : 12.8
# cuDNN           : 9
# Python          : 3.10
# PyTorch         : 2.7.1 + cu128  (native CUDA 12.8 / sm_120 support)
#
# Build context   : Hunyuan3D-Part/ (parent of P3-SAM/)
# Build:
#   cd Hunyuan3D-Part/
#   docker build -t p3sam:latest -f P3-SAM/Dockerfile .
#
# Or use the helper script:
#   ./P3-SAM/docker-build.sh
#
# Run (quick test):
#   docker run --gpus all -p 5001:5001 p3sam:latest
#
# NOTE: First request triggers lazy model download from Hugging Face.
#       Mount volumes at /root/.cache/p3sam and /root/sonata to persist
#       model weights across container restarts (see docker-compose.yml).
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# ── Build-time arguments ──────────────────────────────────────────────────────
# CUDA arch list — sm_120 = RTX Pro 6000 (Blackwell).
# Trim to just "12.0" to speed up builds targeting only this machine.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0"

# Parallel compile jobs for ninja.
ARG MAX_JOBS=4

# ── Proxy (build-time + runtime) ──────────────────────────────────────────────
ARG http_proxy="http://proxy.intra:80"
ARG https_proxy="http://proxy.intra:80"
ARG no_proxy="localhost,127.0.0.1"

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    no_proxy=${no_proxy} \
    NO_PROXY=${no_proxy}

# ── Environment variables ─────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # CUDA
    CUDA_HOME=/usr/local/cuda \
    # Exported so every pip-compiled CUDA extension picks up the right targets
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    # spconv backend
    SPCONV_ALGO=native \
    # HuggingFace — p3sam weights download via hf_hub_download land in /root/.cache/p3sam
    # Sonata weights land in /root/sonata (hardcoded in model.py)
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# ── apt proxy config ──────────────────────────────────────────────────────────
RUN printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' \
      "${http_proxy}" "${https_proxy}" \
      > /etc/apt/apt.conf.d/99proxy

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python runtime + headers (needed to compile C extensions)
    python3.10 \
    python3.10-dev \
    python3-pip \
    # CUDA extension build chain
    build-essential \
    ninja-build \
    cmake \
    git \
    wget \
    curl \
    # OpenGL / EGL (trimesh rendering backend)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # OpenMP (scipy, numba, trimesh)
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.10 as default interpreter ───────────────────────────────────────
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

WORKDIR /app

# =============================================================================
# STEP 1 — PyTorch 2.7.1 with CUDA 12.8 wheels
#
# cu128 wheels match the nvcc 12.8 toolchain on the host.
# Using the native CUDA 12.8 build avoids the ABI mismatch that breaks
# CUDA extension compilation (chamfer3D, spconv) when cu126 wheels are
# used against an nvcc 12.8 environment.
# =============================================================================
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# =============================================================================
# STEP 2 — spconv
#
# spconv-cu120 (CUDA 12.0 ABI) is binary-compatible with the CUDA 12.8
# runtime and is needed by the Sonata feature extractor.
# =============================================================================
RUN pip install --no-cache-dir spconv-cu120==2.3.6

# =============================================================================
# STEP 3 — PyG packages (torch-scatter + extras)
#
# torch-scatter is required by the Sonata model (torch_scatter.segment_csr).
# The full pyg bundle is installed to match the verified local environment.
# Index URL uses torch-2.7.1+cu128 wheels.
# =============================================================================
RUN pip install --no-cache-dir \
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

# =============================================================================
# STEP 4 — flash-attention (build from source — expect ~20-40 min)
#
# Required by the Sonata feature extractor: enable_flash=True by default,
# and the attention layer asserts flash_attn is not None at model init time.
# Must be built against the already-installed torch==2.7.1+cu128.
# MAX_JOBS controls parallel ninja jobs (set low to avoid OOM during build).
# =============================================================================
RUN MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

# =============================================================================
# STEP 5 — Pure-Python / non-CUDA packages
# =============================================================================
COPY P3-SAM/requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt

# =============================================================================
# STEP 6 — Application source code
#
# Build context is the Hunyuan3D-Part/ parent directory.
# We copy only what is needed at runtime:
#   - P3-SAM/demo/      : API entry-point, auto_mask.py
#   - P3-SAM/model.py   : P3SAM model definition (imported as "..")
#   - P3-SAM/utils/     : chamfer3D CUDA extension source
#   - XPart/partgen/    : Sonata feature extractor (local Python package)
# =============================================================================
COPY P3-SAM/demo/    /app/P3-SAM/demo/
COPY P3-SAM/model.py /app/P3-SAM/model.py
COPY P3-SAM/utils/   /app/P3-SAM/utils/
COPY XPart/partgen/  /app/XPart/partgen/

# =============================================================================
# STEP 7 — Build chamfer3D CUDA extension
#
# chamfer3D is a lightweight CUDA kernel used by auto_mask.py.
# Building here ensures sm_120 kernels are compiled into the extension.
# =============================================================================
RUN pip install --no-cache-dir --no-build-isolation /app/P3-SAM/utils/chamfer3D

# =============================================================================
# Runtime working directory
#
# Must be /app/P3-SAM/demo because:
#   - p3sam_api.py does:  from auto_mask import AutoMask
#   - auto_mask.py does:  sys.path.append('..')  →  imports model.py from /app/P3-SAM/
#   - model.py does:      sys.path.append('../XPart/partgen')  →  /app/XPart/partgen
# =============================================================================
WORKDIR /app/P3-SAM/demo

# Create cache mount points (weights are downloaded on first request)
RUN mkdir -p /root/.cache/p3sam /root/sonata /root/.cache/huggingface /app/logs

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 5001

# ── Health check ──────────────────────────────────────────────────────────────
# start-period is long because first request triggers lazy model download
# (P3-SAM ~1 GB + Sonata from HuggingFace).
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=300s \
    --retries=5 \
    CMD curl -f http://localhost:5001/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Single worker: the GPU pipeline loads/unloads model per-request;
# multiple workers would fight for VRAM without benefit.
CMD ["python", "-m", "uvicorn", "p3sam_api:app", \
     "--host", "0.0.0.0", \
     "--port", "5001", \
     "--workers", "1", \
     "--log-level", "info"]
