# V100 validation

## Scope

This run validates the runtime-adjacent SGLang evidence-root path on one
Tesla V100-SXM2-16GB (SM70). It does not validate kernel fusion, CUDA Graph
capture, key custody, or hardware attestation.

Current SGLang and FlashInfer require SM75 or newer. The V100 run therefore
used `launch_torch27_compat.py` with FP16, Torch-native attention, native
SGLang RMSNorm, FlashInfer disabled, and CUDA Graphs disabled. SGLang's
Torch-native sampler returned CPU `int32` token IDs, so the evidence ABI
boundary staged each update onto the current GPU as contiguous `int64`.

## Correctness

- The CUDA provider compiled for `sm_70` with 167 registers, a 432-byte
  stack frame, and no reported spill loads or stores.
- All 13 focused evidence tests passed against the compiled CUDA provider.
- A manual Qwen3-0.6B chat decode emitted GPU-rooted JSONL transcripts; every
  transcript passed canonical verification.
- Both paired benchmarks completed their forced tool-call and tool-result
  probes and verified every emitted transcript.

## Measurements

The two-round smoke run used 8 requests, concurrency 2, and 16 generated
tokens. Median paired throughput loss was 23.69% (p95 24.16%).

The five-round run used 32 requests per mode per round, concurrency 8, four
warmups, and 32 generated tokens:

| Metric | Baseline | Evidence |
| --- | ---: | ---: |
| Median throughput | 67.11 tok/s | 65.56 tok/s |
| Median request latency | 3799.50 ms | 3899.35 ms |
| Median p95 latency | 3830.47 ms | 3924.18 ms |

The median paired throughput loss was 2.76%, with a 4.27% p95. Individual
round losses were -28.51%, 4.27%, 1.99%, 2.76%, and 3.76%; the first round is
a cold/noisy outlier and is retained rather than discarded.

## Interpretation

The larger workload substantially amortized the fixed evidence cost, but did
not meet the sub-1% target. These numbers include an SM70-only CPU-to-GPU token
staging copy and exclude CUDA Graphs, so they are a portability result rather
than the performance verdict for the intended SM80+ decode-adjacent path.

Raw local artifacts:

- `artifacts/v100-paired-full-result.json`
- `artifacts/v100-paired-full.tar.gz`
- Archive SHA-256:
  `6028fde3eaa930a2ef6fb29df676d0f300afc47ee75acedea37807c41a8c265b`
