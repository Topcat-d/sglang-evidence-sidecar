"""Run evidence-off/on SGLang servers and emit a machine-readable comparison."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
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


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def request_once(
    port: int, model: str, index: int, max_tokens: int
) -> dict[str, object]:
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


def post_chat(port: int, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def run_tool_probe(port: int, model: str, max_tokens: int) -> dict[str, object]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_temperature",
                "description": "Return the current temperature for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    user_message = {"role": "user", "content": "What is the temperature in Reno?"}
    first = post_chat(
        port,
        {
            "model": model,
            "messages": [user_message],
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": max(64, max_tokens),
        },
    )
    assistant = first["choices"][0]["message"]
    tool_calls = assistant.get("tool_calls") or []
    if not tool_calls:
        raise RuntimeError("semantic tool probe did not produce a parsed tool call")
    tool_call_id = tool_calls[0]["id"]
    second = post_chat(
        port,
        {
            "model": model,
            "messages": [
                user_message,
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": '{"temperature_f": 72}',
                },
            ],
            "tools": tools,
            "tool_choice": "none",
            "temperature": 0,
            "max_tokens": max_tokens,
        },
    )
    return {
        "tool_call_response_id": first["id"],
        "tool_result_response_id": second["id"],
        "tool_call_id": tool_call_id,
    }


def run_mode(
    args, mode: str, round_index: int, evidence_dir: Path, log_path: Path
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(args.repo / "python") + os.pathsep + env.get("PYTHONPATH", "")
    )
    env.pop("SGLANG_EVIDENCE_SIDECAR_DIR", None)
    env.pop("SGLANG_EVIDENCE_GPU_LIBRARY", None)
    if mode == "evidence":
        env["SGLANG_EVIDENCE_SIDECAR_DIR"] = str(evidence_dir)
        env["SGLANG_EVIDENCE_GPU_LIBRARY"] = str(args.library)
        env["SGLANG_EVIDENCE_MODEL_ID"] = args.model
    if args.disable_flashinfer:
        env["SGLANG_IS_FLASHINFER_AVAILABLE"] = "false"
    command = [sys.executable]
    if args.launcher_script:
        command.append(str(args.launcher_script))
    else:
        command.extend(["-m", "sglang.launch_server"])
    command.extend([
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ])
    if args.tp_size > 1:
        command.extend(["--tp-size", str(args.tp_size)])
    if args.served_model_name:
        command.extend(["--served-model-name", args.served_model_name])
    if args.mem_fraction_static is not None:
        command.extend(["--mem-fraction-static", str(args.mem_fraction_static)])
    if args.tool_call_parser:
        command.extend(["--tool-call-parser", args.tool_call_parser])
    if args.attention_backend:
        command.extend(["--attention-backend", args.attention_backend])
    if args.disable_cuda_graphs:
        command.extend(
            [
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
            ]
        )
    api_model = args.served_model_name or args.model
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            wait_ready(args.port, process, args.startup_timeout)
            for index in range(args.warmup):
                request_once(args.port, api_model, -index - 1, args.max_tokens)
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency
            ) as pool:
                futures = [
                    pool.submit(
                        request_once, args.port, api_model, index, args.max_tokens
                    )
                    for index in range(args.requests)
                ]
                rows = [future.result() for future in futures]
            elapsed = time.perf_counter() - started
            tool_probe = (
                run_tool_probe(args.port, api_model, args.max_tokens)
                if args.tool_call_parser
                else None
            )
        finally:
            stop_server(process)
    latencies = [float(row["latency_ms"]) for row in rows]
    tokens = sum(int(row["completion_tokens"]) for row in rows)
    transcripts = []
    if mode == "evidence":
        labels_by_request: dict[str, set[str]] = {}
        for path in sorted(evidence_dir.glob("*.jsonl")):
            transcripts.append({"path": str(path), **verify_jsonl(path)})
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                labels_by_request.setdefault(str(record["request_id"]), set()).add(
                    str(record["label"])
                )
        if len(transcripts) < args.requests:
            raise RuntimeError(
                f"expected at least {args.requests} verified transcripts, got {len(transcripts)}"
            )
        if tool_probe:
            call_labels = labels_by_request.get(
                str(tool_probe["tool_call_response_id"]), set()
            )
            result_labels = labels_by_request.get(
                str(tool_probe["tool_result_response_id"]), set()
            )
            if "tool_call:sglang_v0" not in call_labels:
                raise RuntimeError(
                    "parsed tool call is missing from its evidence transcript"
                )
            if "tool_result:sglang_v0" not in result_labels:
                raise RuntimeError(
                    "tool result is missing from its evidence transcript"
                )
    return {
        "mode": mode,
        "round": round_index,
        "requests": len(rows),
        "completion_tokens": tokens,
        "elapsed_s": elapsed,
        "throughput_tokens_s": tokens / elapsed,
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": percentile(latencies, 0.95),
        "transcripts": transcripts,
        "tool_probe": tool_probe,
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
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--served-model-name")
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--launcher-script", type=Path)
    parser.add_argument("--attention-backend")
    parser.add_argument("--disable-cuda-graphs", action="store_true")
    parser.add_argument("--disable-flashinfer", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.library = args.library.resolve()
    if args.launcher_script:
        args.launcher_script = args.launcher_script.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.rounds < 2:
        parser.error("--rounds must be at least 2 to reduce server-order bias")
    if args.requests < args.concurrency:
        parser.error("--requests must be greater than or equal to --concurrency")
    if args.tp_size < 1:
        parser.error("--tp-size must be at least 1")

    runs: list[dict[str, object]] = []
    for round_index in range(args.rounds):
        order = (
            ("baseline", "evidence")
            if round_index % 2 == 0
            else ("evidence", "baseline")
        )
        for mode in order:
            run_dir = args.artifact_dir / f"round-{round_index:02d}" / mode
            evidence_dir = run_dir / "transcripts"
            evidence_dir.mkdir(parents=True, exist_ok=False)
            runs.append(
                run_mode(
                    args,
                    mode,
                    round_index,
                    evidence_dir,
                    run_dir / "server.log",
                )
            )

    def aggregate(mode: str) -> dict[str, object]:
        selected = [run for run in runs if run["mode"] == mode]
        throughputs = [float(run["throughput_tokens_s"]) for run in selected]
        medians = [float(run["latency_median_ms"]) for run in selected]
        p95s = [float(run["latency_p95_ms"]) for run in selected]
        return {
            "mode": mode,
            "rounds": len(selected),
            "throughput_tokens_s_median": statistics.median(throughputs),
            "throughput_tokens_s_p95": percentile(throughputs, 0.95),
            "latency_median_ms_median": statistics.median(medians),
            "latency_p95_ms_median": statistics.median(p95s),
        }

    baseline = aggregate("baseline")
    evidence = aggregate("evidence")
    paired_losses = []
    for round_index in range(args.rounds):
        by_mode = {
            str(run["mode"]): run for run in runs if int(run["round"]) == round_index
        }
        baseline_throughput = float(by_mode["baseline"]["throughput_tokens_s"])
        evidence_throughput = float(by_mode["evidence"]["throughput_tokens_s"])
        paired_losses.append(
            {
                "round": round_index,
                "throughput_loss_percent": (
                    (baseline_throughput - evidence_throughput)
                    / baseline_throughput
                    * 100
                ),
            }
        )
    loss_values = [float(row["throughput_loss_percent"]) for row in paired_losses]
    throughput_loss = statistics.median(loss_values)
    result = {
        "schema": "sglang_evidence_small_model_benchmark_v1",
        "model": args.model,
        "measurement": {
            "rounds": args.rounds,
            "requests_per_mode_per_round": args.requests,
            "concurrency": args.concurrency,
            "warmup_requests": args.warmup,
            "max_tokens": args.max_tokens,
            "tp_size": args.tp_size,
            "served_model_name": args.served_model_name or args.model,
            "mem_fraction_static": args.mem_fraction_static,
            "tool_call_parser": args.tool_call_parser,
            "launcher_script": (
                str(args.launcher_script) if args.launcher_script else None
            ),
            "attention_backend": args.attention_backend,
            "cuda_graphs_disabled": args.disable_cuda_graphs,
            "flashinfer_disabled": args.disable_flashinfer,
            "order": "alternating baseline/evidence by round",
        },
        "baseline": baseline,
        "evidence": evidence,
        "runs": runs,
        "paired_throughput_losses": paired_losses,
        "throughput_loss_percent": throughput_loss,
        "throughput_loss_p95_percent": percentile(loss_values, 0.95),
        "under_one_percent_target": throughput_loss < 1.0,
        "trust_boundary": "runtime-adjacent GPU roots; not kernel fusion or hardware attestation",
    }
    output = args.artifact_dir / "result.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
