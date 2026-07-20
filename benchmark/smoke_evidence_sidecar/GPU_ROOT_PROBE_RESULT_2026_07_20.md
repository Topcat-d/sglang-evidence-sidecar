# GPU Evidence Root Probe Result

The standalone provider maintains one persistent root per logical request slot,
generates canonical Smoke ABI v0 batch and token event digests on the GPU, and
advances both roots without a per-token device-to-host copy. Inputs are staged
before timing; one checkpoint copy exports all final states after timing.

Command shape:

```powershell
cmd /c benchmark\smoke_evidence_sidecar\build_gpu_root_probe_windows.bat sm_89
$env:PYTHONPATH="$PWD\python"
uv run python benchmark/smoke_evidence_sidecar/verify_gpu_root_probe.py `
  --probe benchmark/smoke_evidence_sidecar/evidence_root_probe.exe `
  --requests 4 --steps 16 --repeats 31
```

Results:

| GPU | arch | median device ms | p95 device ms | CPU-oracle mismatches |
|---|---:|---:|---:|---:|
| RTX 4070 Ti | sm_89 | 0.680992 | 1.685504 | 0 |
| RTX 3060 | sm_86 | 0.792576 | 0.830464 | 0 |

Each request finishes with sequence `32` and logical token index `16`. Every
final root matches the independent Python ABI oracle. The kernel uses 80
registers, a 432-byte stack frame, and zero spill stores/loads on both targets.

This result proves the device state machine and checkpoint-only export shape.
It does not yet prove a SGLang tensor handoff, decode-throughput overhead, CUDA
Graph behavior, or the under-1% target. The next gate wraps this kernel in a
runtime C API that waits on the inference producer stream from an auxiliary
stream and consumes SGLang's device-resident token tensor directly.
