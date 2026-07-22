# A100 native validation

## Scope

This run validates the runtime-adjacent SGLang evidence path on one
NVIDIA A100-SXM4-40GB (SM80) using native BF16, FlashInfer attention and
sampling, full decode CUDA Graphs, and breakable prefill CUDA Graphs. No V100
compatibility launcher or Torch-native attention fallback was used.

The official `lmsysorg/sglang:latest` image at digest
`sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2`
lagged the checked-out SGLang source. The disposable container was aligned to
the branch requirements with `flashinfer-python==0.6.15.post1`, FlashInfer JIT
in place of the stale 0.6.12 cubin packages, and `sglang-kernel==0.4.5`.

## Correctness

- The CUDA provider compiled for `sm_80` with 128 registers, a 432-byte stack
  frame, and no reported spill loads or stores.
- All 13 focused evidence tests passed against the compiled CUDA provider.
- Native Qwen3-0.6B warmup and timed decode used CUDA Graphs.
- Five alternating baseline/evidence rounds completed their token, batch,
  tool-call, tool-result, checkpoint, and final-root transcript checks.
- Every emitted transcript passed canonical verification.

## Sustained measurement

The five-round run used 32 requests per mode per round, concurrency 8, four
warmups, and 32 generated tokens per request.

| Metric | Baseline | Evidence |
| --- | ---: | ---: |
| Median throughput | 2159.99 tok/s | 1524.46 tok/s |
| Median request latency | 117.70 ms | 167.02 ms |
| Median p95 latency | 120.14 ms | 169.27 ms |

The median paired throughput loss was 29.42%, with a 29.59% p95. Individual
round losses were 29.57%, 29.42%, 29.59%, 29.38%, and 29.10%. The low variance
shows this is a stable critical-path cost, not launch-order noise.

## Cost decomposition

The semantic adapter microbenchmark used 8 requests and 32 decode steps:

- CPU canonical transcript path: 132.22 microseconds per token.
- Disabled adapter dispatch: 0.06 ms added for all 256 events.
- GPU provider enqueue and execution: 81.77 microseconds per 8-request batch,
  measured over 1,000 updates with one final CUDA synchronization.

For the 1,024-token model benchmark, the CPU semantic cost predicts roughly
135 ms while 32 GPU batch updates predict roughly 2.6 ms. The measured added
time was approximately 198 ms. CPU canonicalization, hashing, Python record
construction, and checkpoint serialization therefore dominate the current
29% result; the 128-register GPU root primitive is not the primary bottleneck.

## Decision

Do not spend the next optimization cycle reducing P-256 registers. Move
canonical transcript construction off the scheduler critical path using a
bounded compact event journal and a drain worker. Keep the GPU root update on
the auxiliary CUDA stream, preserve backpressure and cancellation semantics,
and compare drained transcript roots against checkpoint/final GPU roots.

A GPU-root-only benchmark may be added as a diagnostic lower bound, but it is
not a Transcript v0 product and must not be reported as complete attestation.

Raw local artifacts:

- `artifacts/a100-native-full-result.json`
- `artifacts/a100-semantic-cost.json`
- `artifacts/a100-native-full.tar.gz`
- Archive SHA-256:
  `551b5a0ea2dab1cda17b700a7c98185096f36b214cba1a9255b0bf0dbe49d7ac`
