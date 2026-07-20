# Smoke Evidence Sidecar

The experimental Smoke evidence sidecar maps actual SGLang scheduler events to
Smoke Attestation ABI v0 transcripts. It is disabled unless
`SGLANG_EVIDENCE_SIDECAR_DIR` is set before the server process starts.

```bash
export SGLANG_EVIDENCE_SIDECAR_DIR=/var/tmp/sglang-evidence
export SGLANG_EVIDENCE_MODEL_ID=Qwen/Qwen3-0.6B
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
