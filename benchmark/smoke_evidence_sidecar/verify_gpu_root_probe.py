"""Run the CUDA root probe and verify final roots with the CPU ABI oracle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

if sys.platform == "win32":
    python_root = Path(__file__).resolve().parents[2] / "python"
    for name, path in {
        "sglang": python_root / "sglang",
        "sglang.srt": python_root / "sglang" / "srt",
        "sglang.srt.evidence_sidecar": python_root / "sglang" / "srt" / "evidence_sidecar",
    }.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules.setdefault(name, module)

from sglang.srt.evidence_sidecar.abi_v0 import (
    CADENCE_BATCH,
    CADENCE_TOKEN_1,
    EVENT_DECODE_BATCH,
    EVENT_TOKEN_WINDOW,
    ZERO_ROOT,
    CanonicalEventV0,
    compute_chain_root,
    hash_bytes,
    hash_token_ids,
)


def oracle_root(request: int, requests: int, steps: int) -> tuple[int, int, str]:
    root = ZERO_ROOT
    sequence = 0
    token_index = 0
    model_hash = bytes((offset + request) & 0xFF for offset in range(32))
    runtime_hash = bytes((0xA0 + offset + request) & 0xFF for offset in range(32))
    batch_hash = bytes(0x40 + offset for offset in range(32))
    empty_hash = hash_bytes(b"")
    for step in range(steps):
        batch = CanonicalEventV0(
            run_id_lo=0x1000 + request,
            run_id_hi=0x2000 + request,
            sequence_id=sequence,
            model_step=step,
            token_start=token_index,
            token_count=0,
            token_stride=0,
            cadence=CADENCE_BATCH,
            event_type=EVENT_DECODE_BATCH,
            model_id_hash=model_hash,
            output_token_hash=empty_hash,
            sampling_metadata_hash=batch_hash,
            runtime_policy_hash=runtime_hash,
        )
        root = compute_chain_root(
            domain=0x54494C455F524F4F,
            sequence_id=sequence,
            prev_tip=root,
            digest=batch.digest(),
        )
        sequence += 1
        token = 100000 + step * requests + request
        event = CanonicalEventV0(
            run_id_lo=0x1000 + request,
            run_id_hi=0x2000 + request,
            sequence_id=sequence,
            model_step=step + 1,
            token_start=token_index,
            token_count=1,
            token_stride=1,
            cadence=CADENCE_TOKEN_1,
            event_type=EVENT_TOKEN_WINDOW,
            model_id_hash=model_hash,
            output_token_hash=hash_token_ids([token]),
            runtime_policy_hash=runtime_hash,
        )
        root = compute_chain_root(
            domain=0x54494C455F524F4F,
            sequence_id=sequence,
            prev_tip=root,
            digest=event.digest(),
        )
        sequence += 1
        token_index += 1
    return sequence, token_index, root.hex()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--api-probe", type=Path)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    completed = subprocess.run(
        [str(args.probe), str(args.requests), str(args.steps), str(args.repeats)],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics: dict[str, object] = {}
    roots: dict[int, tuple[int, int, str]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split(",")
        if fields[0] == "probe":
            metrics[fields[1]] = fields[2]
        elif fields[0] == "root":
            roots[int(fields[1])] = (int(fields[2]), int(fields[3]), fields[4])
    expected = {
        request: oracle_root(request, args.requests, args.steps)
        for request in range(args.requests)
    }
    mismatches = [request for request in expected if roots.get(request) != expected[request]]
    api_roots: dict[int, tuple[int, int, str]] = {}
    api_markers: dict[str, str] = {}
    if args.api_probe:
        api_completed = subprocess.run(
            [str(args.api_probe)], check=True, capture_output=True, text=True
        )
        for line in api_completed.stdout.splitlines():
            fields = line.split(",")
            if fields[0] == "api_root":
                api_roots[int(fields[1])] = (int(fields[2]), int(fields[3]), fields[4])
            elif fields[0] == "api_probe":
                api_markers[fields[1]] = fields[2]
        mismatches.extend(
            request
            for request in expected
            if api_roots.get(request) != expected[request]
        )
    result = {
        "schema": "sglang_gpu_evidence_root_probe_v0",
        "ok": not mismatches,
        "requests": args.requests,
        "steps": args.steps,
        "repeats": args.repeats,
        "mismatches": mismatches,
        "metrics": metrics,
        "roots": {str(key): value[2] for key, value in roots.items()},
        "api_roots": {str(key): value[2] for key, value in api_roots.items()},
        "api_markers": api_markers,
    }
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
