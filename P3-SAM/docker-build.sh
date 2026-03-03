#!/usr/bin/env bash
# =============================================================================
# P3-SAM API — Docker build helper
#
# Usage:
#   ./P3-SAM/docker-build.sh [TAG] [MAX_JOBS] [CUDA_ARCH_LIST]
#
# Examples:
#   ./P3-SAM/docker-build.sh                        # p3sam:latest, 4 jobs, all arches
#   ./P3-SAM/docker-build.sh p3sam:v1               # custom tag
#   ./P3-SAM/docker-build.sh p3sam:latest 8         # 8 parallel compile jobs
#   ./P3-SAM/docker-build.sh p3sam:latest 4 "12.0"  # only Blackwell (faster build)
#
# Proxy (corporate network):
#   http_proxy=http://proxy.intra:80 ./P3-SAM/docker-build.sh
# =============================================================================
set -euo pipefail

TAG="${1:-p3sam:latest}"
MAX_JOBS="${2:-4}"
CUDA_ARCH_LIST="${3:-8.0;8.6;8.9;9.0;10.0;12.0}"

# The Dockerfile COPYs both P3-SAM/ and XPart/, so the build context must be
# the Hunyuan3D-Part/ parent directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=================================================="
echo "  Building P3-SAM Docker image"
echo "  Tag           : ${TAG}"
echo "  Context       : ${CONTEXT_DIR}"
echo "  MAX_JOBS      : ${MAX_JOBS}"
echo "  CUDA arches   : ${CUDA_ARCH_LIST}"
echo "=================================================="

docker build \
    --tag "${TAG}" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --build-arg TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --build-arg http_proxy="${http_proxy:-}" \
    --build-arg https_proxy="${https_proxy:-}" \
    --build-arg no_proxy="${no_proxy:-localhost,127.0.0.1}" \
    "${CONTEXT_DIR}"

echo ""
echo "✅  Build complete: ${TAG}"
echo ""
echo "Run the container:"
echo "  docker run --gpus all -p 5001:5001 \\"
echo "    -v \$(pwd)/P3-SAM/weights/p3sam:/root/.cache/p3sam \\"
echo "    -v \$(pwd)/P3-SAM/weights/sonata:/root/sonata \\"
echo "    ${TAG}"
echo ""
echo "Or use docker compose (from Hunyuan3D-Part/ directory):"
echo "  docker compose -f P3-SAM/docker-compose.yml up -d"
