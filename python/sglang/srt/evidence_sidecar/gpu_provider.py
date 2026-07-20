"""ctypes bridge to the optional CUDA evidence-root runtime."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class RequestStateV0(ctypes.Structure):
    _fields_ = [
        ("run_id_lo", ctypes.c_uint64),
        ("run_id_hi", ctypes.c_uint64),
        ("sequence_id", ctypes.c_uint64),
        ("model_step", ctypes.c_uint64),
        ("logical_token_index", ctypes.c_uint64),
        ("model_id_hash", ctypes.c_uint8 * 32),
        ("runtime_policy_hash", ctypes.c_uint8 * 32),
        ("root", ctypes.c_uint8 * 32),
    ]


class UpdateV0(ctypes.Structure):
    _fields_ = [
        ("slot", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("batch_metadata_hash", ctypes.c_uint8 * 32),
    ]


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    sequence_id: int
    model_step: int
    logical_token_index: int
    root: bytes


def _fill(target, value: bytes) -> None:
    if len(value) != len(target):
        raise ValueError("GPU evidence hash must be 32 bytes")
    target[:] = value


class GpuEvidenceProvider:
    def __init__(self, library: Path, *, max_slots: int, max_updates: int) -> None:
        self._library = ctypes.CDLL(str(library))
        self._configure_signatures()
        self._context = ctypes.c_void_p()
        self._check(
            self._library.sglang_evidence_create_v0(
                max_slots, max_updates, ctypes.byref(self._context)
            ),
            "create",
        )
        self.max_slots = max_slots
        self.max_updates = max_updates

    def _configure_signatures(self) -> None:
        lib = self._library
        lib.sglang_evidence_create_v0.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.sglang_evidence_initialize_slot_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(RequestStateV0),
        ]
        lib.sglang_evidence_update_tokens_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(UpdateV0),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        lib.sglang_evidence_checkpoint_v0.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(RequestStateV0),
        ]
        lib.sglang_evidence_release_slot_v0.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.sglang_evidence_destroy_v0.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"GPU evidence {operation} failed with CUDA status {status}")

    def initialize(
        self,
        slot: int,
        *,
        run_id_lo: int,
        run_id_hi: int,
        sequence_id: int,
        model_step: int,
        logical_token_index: int,
        model_id_hash: bytes,
        runtime_policy_hash: bytes,
        root: bytes,
    ) -> None:
        state = RequestStateV0()
        state.run_id_lo = run_id_lo
        state.run_id_hi = run_id_hi
        state.sequence_id = sequence_id
        state.model_step = model_step
        state.logical_token_index = logical_token_index
        _fill(state.model_id_hash, model_id_hash)
        _fill(state.runtime_policy_hash, runtime_policy_hash)
        _fill(state.root, root)
        self._check(
            self._library.sglang_evidence_initialize_slot_v0(
                self._context, slot, ctypes.byref(state)
            ),
            "initialize",
        )

    def update(self, rows: Iterable[tuple[int, bytes]], token_tensor) -> None:
        import torch

        values = list(rows)
        if not values or len(values) > self.max_updates:
            raise ValueError("GPU evidence update count is invalid")
        if not token_tensor.is_cuda or token_tensor.dtype != torch.int64:
            raise ValueError("GPU evidence tokens must be a CUDA int64 tensor")
        if not token_tensor.is_contiguous() or token_tensor.numel() != len(values):
            raise ValueError("GPU evidence token tensor must be contiguous and match updates")
        updates = (UpdateV0 * len(values))()
        for index, (slot, metadata_hash) in enumerate(values):
            updates[index].slot = slot
            _fill(updates[index].batch_metadata_hash, metadata_hash)
        stream = torch.cuda.current_stream(token_tensor.device).cuda_stream
        self._check(
            self._library.sglang_evidence_update_tokens_v0(
                self._context,
                updates,
                ctypes.c_void_p(token_tensor.data_ptr()),
                len(values),
                ctypes.c_void_p(stream),
            ),
            "update",
        )

    def checkpoint(self, slots: Iterable[int]) -> list[StateSnapshot]:
        values = list(slots)
        slot_array = (ctypes.c_uint32 * len(values))(*values)
        states = (RequestStateV0 * len(values))()
        self._check(
            self._library.sglang_evidence_checkpoint_v0(
                self._context, slot_array, len(values), states
            ),
            "checkpoint",
        )
        return [
            StateSnapshot(
                sequence_id=state.sequence_id,
                model_step=state.model_step,
                logical_token_index=state.logical_token_index,
                root=bytes(state.root),
            )
            for state in states
        ]

    def release(self, slot: int) -> None:
        self._check(
            self._library.sglang_evidence_release_slot_v0(self._context, slot),
            "release",
        )

    def close(self) -> None:
        if getattr(self, "_context", None):
            self._library.sglang_evidence_destroy_v0(self._context)
            self._context = ctypes.c_void_p()

    def __del__(self) -> None:
        self.close()
