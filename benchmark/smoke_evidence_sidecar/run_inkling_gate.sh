#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${MODEL:-thinkingmachines/Inkling-NVFP4}
TP_SIZE=${TP_SIZE:-8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-inkling}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-inkling}
if [[ -z "${MIN_TOTAL_VRAM_GIB:-}" ]]; then
  if [[ "$MODEL" == *Inkling-NVFP4 ]]; then
    MIN_TOTAL_VRAM_GIB=700
  else
    MIN_TOTAL_VRAM_GIB=2100
  fi
fi
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$ROOT/artifacts/evidence-inkling-$RUN_ID"}

export MODEL TP_SIZE SERVED_MODEL_NAME MEM_FRACTION_STATIC TOOL_CALL_PARSER
export RUN_ID ARTIFACT_ROOT
export REQUESTS=${REQUESTS:-64}
export CONCURRENCY=${CONCURRENCY:-16}
export MAX_TOKENS=${MAX_TOKENS:-128}
export ROUNDS=${ROUNDS:-5}
export STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-3600}

python - "$MODEL" "$TP_SIZE" "$MIN_TOTAL_VRAM_GIB" <<'PY'
import sys
import torch

model, tp_size_raw, minimum_raw = sys.argv[1:]
tp_size = int(tp_size_raw)
minimum = float(minimum_raw)
visible = torch.cuda.device_count()
if visible < tp_size:
    raise SystemExit(
        f"Inkling requires TP_SIZE={tp_size}, but only {visible} CUDA GPUs are visible"
    )

selected = list(range(tp_size))
total_gib = sum(
    torch.cuda.get_device_properties(index).total_memory for index in selected
) / 2**30
capabilities = [torch.cuda.get_device_capability(index) for index in selected]
names = [torch.cuda.get_device_name(index) for index in selected]
if total_gib < minimum:
    raise SystemExit(
        f"refusing Inkling download: selected GPUs provide {total_gib:.1f} GiB, "
        f"below MIN_TOTAL_VRAM_GIB={minimum:.1f}"
    )
if model.endswith("Inkling-NVFP4") and any(
    capability < (10, 0) for capability in capabilities
):
    raise SystemExit(
        "thinkingmachines/Inkling-NVFP4 requires Blackwell-class CUDA capability 10.0+"
    )
print(
    {
        "model": model,
        "gpu_names": names,
        "capabilities": capabilities,
        "total_gib": total_gib,
    }
)
PY

exec "$ROOT/benchmark/smoke_evidence_sidecar/run_runpod_small_model_gate.sh"
