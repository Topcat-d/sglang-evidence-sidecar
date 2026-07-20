#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARCH=${ARCH:-sm_80}
SOURCE="$ROOT/sgl-kernel/csrc/evidence_sidecar/evidence_root_api.cu"
OUTPUT=${OUTPUT:-"$ROOT/benchmark/smoke_evidence_sidecar/libsglang_evidence_root.so"}

nvcc -std=c++17 -O3 -lineinfo -Xptxas=-v -arch="$ARCH" \
  -Xcompiler=-fPIC -shared "$SOURCE" -o "$OUTPUT"
printf '%s\n' "$OUTPUT"
