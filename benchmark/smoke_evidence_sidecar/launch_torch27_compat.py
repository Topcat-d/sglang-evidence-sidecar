"""Launch current SGLang on Torch 2.7 for the V100 validation environment."""

from __future__ import annotations

import runpy

import flashinfer.fp4_quantization as fp4_quantization
import flashinfer.prefill as flashinfer_prefill
import torch
import torch.cuda.memory as cuda_memory


if not hasattr(cuda_memory, "_cuda_beginAllocateCurrentThreadToPool"):
    cuda_memory._cuda_beginAllocateCurrentThreadToPool = (  # type: ignore[attr-defined]
        cuda_memory._cuda_beginAllocateToPool
    )
if not hasattr(cuda_memory, "_cuda_endAllocateToPool"):
    cuda_memory._cuda_endAllocateToPool = (  # type: ignore[attr-defined]
        cuda_memory._cuda_endAllocateCurrentStreamToPool
    )


def _unsupported_fp4(*_args, **_kwargs):
    raise RuntimeError("FP4 operations are unavailable in the V100 validation image")


if not hasattr(fp4_quantization, "block_scale_interleave"):
    fp4_quantization.block_scale_interleave = _unsupported_fp4
if not hasattr(flashinfer_prefill, "cudnn_batch_prefill_with_kv_cache"):
    flashinfer_prefill.cudnn_batch_prefill_with_kv_cache = _unsupported_fp4


def _allow_sm70_fp16(*, server_args, model_config):
    """Retain SGLang's FP16 fallback without its SM75 policy guard."""
    if torch.cuda.get_device_capability()[0] < 8:
        from sglang.srt.arg_groups.overrides import declare_load_time_override

        declare_load_time_override(
            "ModelRunner._sm80_dtype_fallback", {"dtype": "float16"}
        )
        model_config.dtype = torch.float16


from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.layers.layernorm import RMSNorm

load_model_utils.maybe_downgrade_dtype_for_legacy_gpu = _allow_sm70_fp16
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7:
    RMSNorm.forward_cuda = RMSNorm.forward_native

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")
