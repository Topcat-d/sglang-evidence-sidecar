#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
REQUESTS=${REQUESTS:-32}
CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-32}
ROUNDS=${ROUNDS:-5}
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
  printf 'run_id=%s\nmodel=%s\narch=%s\nrequests=%s\nconcurrency=%s\nmax_tokens=%s\nrounds=%s\n' \
    "$RUN_ID" "$MODEL" "$ARCH" "$REQUESTS" "$CONCURRENCY" "$MAX_TOKENS" "$ROUNDS"
  git rev-parse HEAD
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  nvcc --version
} > "$ARTIFACT_ROOT/environment.txt"

python -m pip install --no-deps -e python
benchmark/smoke_evidence_sidecar/build_gpu_root_provider_posix.sh
python -m pytest test/srt/test_evidence_sidecar.py -q

python benchmark/smoke_evidence_sidecar/run_small_model_benchmark.py \
  --repo "$ROOT" \
  --model "$MODEL" \
  --library "$LIBRARY" \
  --artifact-dir "$ARTIFACT_ROOT/benchmark" \
  --requests "$REQUESTS" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --rounds "$ROUNDS"

ARCHIVE="$ARTIFACT_ROOT.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$ARTIFACT_ROOT")" "$(basename "$ARTIFACT_ROOT")"
printf 'gate complete\nresult=%s\narchive=%s\n' \
  "$ARTIFACT_ROOT/benchmark/result.json" "$ARCHIVE"
