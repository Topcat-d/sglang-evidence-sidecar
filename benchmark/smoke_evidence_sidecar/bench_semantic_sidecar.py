"""Measure Python semantic-sidecar cost before GPU-root integration."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import types
import uuid
from dataclasses import dataclass
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

from sglang.srt.evidence_sidecar.runtime import (
    EvidenceSink,
    evidence_batch,
    evidence_tokens,
)


@dataclass(slots=True)
class BenchReq:
    rid: str
    session_id: str


def _baseline(requests: list[BenchReq], steps: int) -> int:
    checksum = 0
    for step in range(steps):
        for index, _req in enumerate(requests):
            checksum ^= (step << 8) ^ index
    return checksum


def _evidence(output: Path, requests: list[BenchReq], steps: int) -> int:
    sink = EvidenceSink(output / uuid.uuid4().hex, model_id="semantic-benchmark")
    checksum = 0
    for step in range(steps):
        sink.record_batch(requests, batch_kind="decode")
        for index, req in enumerate(requests):
            token = (step << 8) ^ index
            sink.record_tokens(req, [token], source="decode")
            checksum ^= token
    for req in requests:
        sink.close(req, reason="completed")
    return checksum


def _disabled(requests: list[BenchReq], steps: int) -> int:
    checksum = 0
    for step in range(steps):
        evidence_batch(requests, batch_kind="decode")
        for index, req in enumerate(requests):
            token = (step << 8) ^ index
            evidence_tokens(req, [token], source="decode")
            checksum ^= token
    return checksum


def _measure(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if min(args.requests, args.steps, args.repeats) <= 0:
        parser.error("requests, steps, and repeats must be positive")

    requests = [BenchReq(f"request-{i}", f"session-{i}") for i in range(args.requests)]
    with tempfile.TemporaryDirectory(prefix="sglang-evidence-bench-") as temporary:
        output = Path(temporary)
        baseline = _measure(lambda: _baseline(requests, args.steps), args.repeats)
        disabled = _measure(lambda: _disabled(requests, args.steps), args.repeats)
        evidence = _measure(lambda: _evidence(output, requests, args.steps), args.repeats)

    baseline_median = statistics.median(baseline)
    evidence_median = statistics.median(evidence)
    disabled_median = statistics.median(disabled)
    events = args.requests * args.steps
    result = {
        "schema": "sglang_evidence_semantic_benchmark_v0",
        "requests": args.requests,
        "steps": args.steps,
        "events": events,
        "repeats": args.repeats,
        "baseline_ms": baseline,
        "evidence_ms": evidence,
        "disabled_ms": disabled,
        "baseline_median_ms": baseline_median,
        "evidence_median_ms": evidence_median,
        "disabled_median_ms": disabled_median,
        "disabled_added_median_ms": disabled_median - baseline_median,
        "added_median_ms": evidence_median - baseline_median,
        "evidence_us_per_token": evidence_median * 1000 / events,
        "scope": "python_semantic_adapter_not_model_decode_or_gpu_root",
    }
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
