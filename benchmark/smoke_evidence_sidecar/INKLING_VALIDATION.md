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

### Single-A100 preflight result

The July 22, 2026 native validation host exposed one A100-SXM4-40GB. It was
used successfully for the Qwen3-0.6B evidence benchmark, but Inkling was
rejected before weight download. One 40 GiB device cannot hold either the
roughly 600 GB NVFP4 checkpoint or the roughly 1.9 TB BF16 checkpoint, and
SM80 cannot execute the intended Blackwell NVFP4 path. CPU or disk offload
would test a different, transfer-dominated system and is not a valid
colocation performance result.

See `A100_VALIDATION.md` for the native FlashInfer/CUDA Graph baseline and the
measured evidence-cost decomposition.

## Colocation requirements

For this experiment, *colocated* means the evidence producer executes inside
the same SGLang scheduler process and CUDA device context as the accepted
decode tokens. The provider may use a nonblocking auxiliary CUDA stream, but
it must consume token IDs directly from the owning rank's device memory using
CUDA event ordering. It must not receive tokens through HTTP, a remote queue,
or a post-response log scraper.

Inkling's model weights are distributed across a large GPU pod. Preserve these
placement rules:

- Keep all model ranks in one NVLink/NVSwitch domain where the supported
  Inkling recipe expects it. Do not compare a tightly packed baseline with an
  evidence run spread across a different network topology.
- Start one evidence-provider context per owning CUDA device. Do not share a
  raw CUDA context, stream, or device pointer across processes or ranks.
- Emit one logical transcript per request. Only the attention-parallel leader
  that owns the accepted token stream updates that request's evidence state;
  tensor, expert, and pipeline replicas must not create duplicate chains.
- Bind the accepted token tensor to the auxiliary stream with producer and
  consumer CUDA events. A host copy followed by a new device upload is a
  compatibility fallback, not the target colocated path.
- Read back only checkpoint or final roots. Per-token stream synchronization
  or device-to-host root copies invalidate the launch-overhead hypothesis.
- Reserve evidence memory identically in baseline and evidence runs, or report
  the difference. Keep `MEM_FRACTION_STATIC`, tensor/expert parallel settings,
  CUDA Graph sizes, scheduler policy, and request order fixed between modes.
- Use local or pod-attached storage for model weights and transcript artifacts.
  Weight-download, cold JIT, graph-capture, and network-storage time must remain
  outside the timed decode interval.
- Do not use MIG for the primary comparison unless both modes use the same MIG
  profile and topology. Report GPU UUIDs and rank-to-device placement.

For orchestration systems, use strict/packed placement rather than allowing
ranks to drift across hosts. Capture the host list, rank map, GPU UUIDs,
NVLink/NVSwitch topology, SGLang commit, image digest, CUDA/driver versions,
FlashInfer and `sglang-kernel` versions, model revision, and quantization
revision in the artifact bundle.

### What does not count

The following can validate transcript semantics, but not colocation or
kernel-adjacency performance:

- Calling a hosted Inkling API and signing returned text or token IDs.
- Running the evidence provider on a different GPU or host from decode.
- Reconstructing evidence only from HTTP responses or server logs.
- Using CPU/disk weight offload to force Inkling onto an undersized pod.
- Reporting GPU-root-only timing as complete Transcript v0 attestation.

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
attestation, or resistance to a malicious host. Physical colocation reduces
the observation gap between decode and evidence generation; it does not by
itself prove that the host, scheduler binary, model weights, or GPU identity
are trustworthy.
