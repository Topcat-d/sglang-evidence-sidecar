"""Run evidence-off/on SGLang servers and emit a machine-readable comparison."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from sglang.srt.evidence_sidecar.abi_v0 import verify_jsonl


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def wait_ready(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError("SGLang server did not become healthy")


def request_once(port: int, model: str, index: int, max_tokens: int) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Count from one to five, then print request-{index}.",
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "completion_tokens": int(body["usage"]["completion_tokens"]),
        "id": body["id"],
        "latency_ms": latency_ms,
    }


def run_mode(args, mode: str, evidence_dir: Path, log_path: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo / "python") + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SGLANG_EVIDENCE_SIDECAR_DIR", None)
    env.pop("SGLANG_EVIDENCE_GPU_LIBRARY", None)
    if mode == "evidence":
        env["SGLANG_EVIDENCE_SIDECAR_DIR"] = str(evidence_dir)
        env["SGLANG_EVIDENCE_GPU_LIBRARY"] = str(args.library)
        env["SGLANG_EVIDENCE_MODEL_ID"] = args.model
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_ready(args.port, process, args.startup_timeout)
            for index in range(args.warmup):
                request_once(args.port, args.model, -index - 1, args.max_tokens)
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [
                    pool.submit(request_once, args.port, args.model, index, args.max_tokens)
                    for index in range(args.requests)
                ]
                rows = [future.result() for future in futures]
            elapsed = time.perf_counter() - started
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    latencies = [float(row["latency_ms"]) for row in rows]
    tokens = sum(int(row["completion_tokens"]) for row in rows)
    transcripts = []
    if mode == "evidence":
        for path in sorted(evidence_dir.glob("*.jsonl")):
            transcripts.append({"path": str(path), **verify_jsonl(path)})
        if len(transcripts) < args.requests:
            raise RuntimeError(
                f"expected at least {args.requests} verified transcripts, got {len(transcripts)}"
            )
    return {
        "mode": mode,
        "requests": len(rows),
        "completion_tokens": tokens,
        "elapsed_s": elapsed,
        "throughput_tokens_s": tokens / elapsed,
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "transcripts": transcripts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.library = args.library.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.artifact_dir / "transcripts"
    evidence_dir.mkdir(exist_ok=True)

    baseline = run_mode(args, "baseline", evidence_dir, args.artifact_dir / "baseline.log")
    evidence = run_mode(args, "evidence", evidence_dir, args.artifact_dir / "evidence.log")
    throughput_loss = (
        (float(baseline["throughput_tokens_s"]) - float(evidence["throughput_tokens_s"]))
        / float(baseline["throughput_tokens_s"])
        * 100
    )
    result = {
        "schema": "sglang_evidence_small_model_benchmark_v0",
        "model": args.model,
        "baseline": baseline,
        "evidence": evidence,
        "throughput_loss_percent": throughput_loss,
        "under_one_percent_target": throughput_loss < 1.0,
        "trust_boundary": "runtime-adjacent GPU roots; not kernel fusion or hardware attestation",
    }
    output = args.artifact_dir / "result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
