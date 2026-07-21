#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
REQUESTS=${REQUESTS:-32}
CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-32}
ROUNDS=${ROUNDS:-5}
TP_SIZE=${TP_SIZE:-1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-600}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen}
LAUNCHER_SCRIPT=${LAUNCHER_SCRIPT:-}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-}
DISABLE_CUDA_GRAPHS=${DISABLE_CUDA_GRAPHS:-false}
DISABLE_FLASHINFER=${DISABLE_FLASHINFER:-false}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$ROOT/artifacts/evidence-small-model-$RUN_ID"}
LIBRARY="$ROOT/benchmark/smoke_evidence_sidecar/libsglang_evidence_root.so"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc is required; use a RunPod CUDA development image" >&2; exit 2; }
command -v python >/dev/null || { echo "python is required" >&2; exit 2; }

mkdir -p "$ARTIFACT_ROOT"
exec > >(tee "$ARTIFACT_ROOT/gate.log") 2>&1

cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
try:
    import sglang
    import torch
except ImportError as exc:
    raise SystemExit(
        f"missing SGLang runtime dependency ({exc}); start from an SGLang RunPod image "
        "or install SGLang runtime requirements before running this gate"
    ) from exc
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU")
print(f"sglang={sglang.__file__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY

if [[ -z "${ARCH:-}" ]]; then
  ARCH=$(python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"sm_{major}{minor}")
PY
  )
fi
export ARCH

{
  printf 'run_id=%s\nmodel=%s\narch=%s\nrequests=%s\nconcurrency=%s\nmax_tokens=%s\nrounds=%s\ntp_size=%s\n' \
    "$RUN_ID" "$MODEL" "$ARCH" "$REQUESTS" "$CONCURRENCY" "$MAX_TOKENS" "$ROUNDS" "$TP_SIZE"
  git rev-parse HEAD
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  nvcc --version
} > "$ARTIFACT_ROOT/environment.txt"

python -m pip install --no-deps -e python
benchmark/smoke_evidence_sidecar/build_gpu_root_provider_posix.sh
python -m pytest test/srt/test_evidence_sidecar.py -q

BENCHMARK_ARGS=(
  --repo "$ROOT"
  --model "$MODEL"
  --library "$LIBRARY"
  --artifact-dir "$ARTIFACT_ROOT/benchmark"
  --requests "$REQUESTS"
  --concurrency "$CONCURRENCY"
  --max-tokens "$MAX_TOKENS"
  --rounds "$ROUNDS"
  --tp-size "$TP_SIZE"
  --startup-timeout "$STARTUP_TIMEOUT"
)
if [[ -n "$SERVED_MODEL_NAME" ]]; then
  BENCHMARK_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [[ -n "$MEM_FRACTION_STATIC" ]]; then
  BENCHMARK_ARGS+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
fi
if [[ -n "$TOOL_CALL_PARSER" ]]; then
  BENCHMARK_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
fi
if [[ -n "$LAUNCHER_SCRIPT" ]]; then
  BENCHMARK_ARGS+=(--launcher-script "$LAUNCHER_SCRIPT")
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
  BENCHMARK_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ "$DISABLE_CUDA_GRAPHS" == true ]]; then
  BENCHMARK_ARGS+=(--disable-cuda-graphs)
fi
if [[ "$DISABLE_FLASHINFER" == true ]]; then
  BENCHMARK_ARGS+=(--disable-flashinfer)
fi
python benchmark/smoke_evidence_sidecar/run_small_model_benchmark.py \
  "${BENCHMARK_ARGS[@]}"

ARCHIVE="$ARTIFACT_ROOT.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$ARTIFACT_ROOT")" "$(basename "$ARTIFACT_ROOT")"
printf 'gate complete\nresult=%s\narchive=%s\n' \
  "$ARTIFACT_ROOT/benchmark/result.json" "$ARCHIVE"
