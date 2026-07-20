#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-0.6B}
PORT=${PORT:-30000}
OUT=${OUT:-"$PWD/evidence-small-model"}
RESULT=${RESULT:-"$PWD/small-model-response.json"}

export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_EVIDENCE_SIDECAR_DIR="$OUT"
export SGLANG_EVIDENCE_MODEL_ID="$MODEL"
export SGLANG_EVIDENCE_GPU_LIBRARY=${SGLANG_EVIDENCE_GPU_LIBRARY:-"$PWD/benchmark/smoke_evidence_sidecar/libsglang_evidence_root.so"}

python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly: evidence-ready\"}],\"temperature\":0,\"max_tokens\":16}" \
  >"$RESULT"

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
from sglang.srt.evidence_sidecar.abi_v0 import verify_jsonl

paths = sorted(Path(sys.argv[1]).glob("*.jsonl"))
if not paths:
    raise SystemExit("no evidence transcript produced")
for path in paths:
    print(json.dumps({"path": str(path), **verify_jsonl(path)}, sort_keys=True))
PY
