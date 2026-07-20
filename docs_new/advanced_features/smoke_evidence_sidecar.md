# Smoke Evidence Sidecar

The experimental Smoke evidence sidecar maps actual SGLang scheduler events to
Smoke Attestation ABI v0 transcripts. It is disabled unless
`SGLANG_EVIDENCE_SIDECAR_DIR` is set before the server process starts.

```bash
export SGLANG_EVIDENCE_SIDECAR_DIR=/var/tmp/sglang-evidence
export SGLANG_EVIDENCE_MODEL_ID=Qwen/Qwen3-0.6B
export SGLANG_EVIDENCE_GPU_LIBRARY=$PWD/benchmark/smoke_evidence_sidecar/libsglang_evidence_root.so
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B
```

The scheduler records accepted prefill and decode token IDs at the point where
they are committed to `Req.output_ids`. This includes all accepted speculative
tokens rather than only tokens later coalesced for streaming. Each request has
an independent logical chain, so continuous-batch reordering and physical slot
reuse do not change its `sequence_id` or `logical_token_index`.

Records stay in memory during decode. Periodic and completion checkpoints write
JSONL atomically. The OpenAI response process resumes the checkpoint by stable
request ID, commits normalized parsed tool calls, and writes the terminal run
boundary. Raw prompts, generated text, and tool payloads are not written; their
canonical hashes feed the descriptor digest.

## Verification

The JSONL format is byte-compatible with the existing Smoke Transcript v0 C++
verifier. The Python verifier is available as
`sglang.srt.evidence_sidecar.abi_v0.verify_jsonl`.

Run focused tests:

```bash
PYTHONPATH=python python -m pytest test/srt/test_evidence_sidecar.py -q
```

Run the semantic-adapter microbenchmark:

```bash
PYTHONPATH=python python benchmark/smoke_evidence_sidecar/bench_semantic_sidecar.py \
  --requests 8 --steps 128 --repeats 7 --json-out semantic-benchmark.json
```

This benchmark measures Python event construction and checkpoint export only.
It does not establish model decode overhead or GPU evidence-root performance.

## Trust Boundary

This phase proves that scheduler-owned token events can be committed into a
continuous, independently verifiable transcript. Roots are currently computed
on the CPU. It does not prove kernel fusion, protected key custody, GPU hardware
identity, remote attestation, or resistance to a malicious host. The next gate
replaces CPU root computation with an auxiliary-stream GPU provider while
retaining the same transcript and verifier semantics.

The standalone CUDA provider and cross-oracle probe live under
`sgl-kernel/csrc/evidence_sidecar` and
`benchmark/smoke_evidence_sidecar/verify_gpu_root_probe.py`. Their result is a
primitive gate only; it must not be cited as SGLang model-overhead evidence.

On Linux, build the provider and run the evidence-off/on small-model gate:

```bash
ARCH=sm_80 benchmark/smoke_evidence_sidecar/build_gpu_root_provider_posix.sh
PYTHONPATH=python python benchmark/smoke_evidence_sidecar/run_small_model_benchmark.py \
  --library benchmark/smoke_evidence_sidecar/libsglang_evidence_root.so \
  --artifact-dir artifacts/evidence-small-model
```

The comparison reports sustained completion-token throughput, median latency,
p95 latency, throughput loss, and verifier results for every evidence-on
transcript. A result above one percent throughput loss is reported as a failed
performance target, not hidden by the correctness result.
