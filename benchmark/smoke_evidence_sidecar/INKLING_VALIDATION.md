# Inkling Evidence Validation

This is the second model gate. Run the unchanged evidence adapter and paired
benchmark only after the small-model gate passes on the same branch.

## Hardware gate

Inkling has 975B total parameters and 41B active parameters. The official BF16
checkpoint occupies about 1.9 TB, while the NVFP4 checkpoint requires roughly
600 GB of aggregate VRAM and Blackwell-class GPUs. An ordinary eight-GPU
H100/H200 pod is therefore not a valid target for the default NVFP4 run.

Primary references:

- [Thinking Machines Inkling model card](https://huggingface.co/thinkingmachines/Inkling)
- [Hugging Face Inkling release and inference requirements](https://huggingface.co/blog/thinkingmachines-inkling)
- [NVFP4 checkpoint](https://huggingface.co/thinkingmachines/Inkling-NVFP4)

The default preflight requires eight visible GPUs, CUDA capability 10.0 or
newer, and at least 700 GiB aggregate VRAM before model download begins. The
extra memory above the published checkpoint size leaves room for runtime state
and KV cache; it is a conservative experiment gate, not a vendor guarantee.
Selecting the BF16 model changes the default aggregate-memory gate to 2100 GiB.

## Run

From a CUDA development image containing the current SGLang runtime:

```bash
benchmark/smoke_evidence_sidecar/run_inkling_gate.sh
```

Useful explicit settings:

```bash
TP_SIZE=8 MEM_FRACTION_STATIC=0.85 \
REQUESTS=64 CONCURRENCY=16 ROUNDS=5 MAX_TOKENS=128 \
benchmark/smoke_evidence_sidecar/run_inkling_gate.sh
```

The script calls `run_runpod_small_model_gate.sh`, which builds the identical
GPU provider, runs focused tests, alternates evidence-off/on server order, and
verifies every transcript. An untimed forced tool exchange requires actual
parsed `tool_call` and historical `tool_result` events in each evidence round.
It records the model, tensor-parallel size, hardware, per-round results, paired
throughput losses, logs, and transcript artifacts.

## Trust boundary

This run demonstrates runtime-adjacent GPU evidence roots over actual Inkling
decode events. Tensor-parallel ranks observe the same accepted token IDs, so
only each attention-parallel leader owns a logical transcript and GPU root.
This does not establish kernel fusion, key custody, GPU identity, remote
attestation, or resistance to a malicious host.
