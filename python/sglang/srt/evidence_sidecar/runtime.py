"""Opt-in scheduler adapter for per-request Smoke evidence transcripts."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from sglang.srt.evidence_sidecar.abi_v0 import (
    CADENCE_ADAPTIVE_CHECKPOINT,
    CADENCE_BATCH,
    CADENCE_RUN_BOUNDARY,
    CADENCE_TOKEN_1,
    CADENCE_TOOL_CALL,
    EVENT_CHECKPOINT,
    EVENT_DECODE_BATCH,
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_TOKEN_WINDOW,
    EVENT_TOOL_CALL,
    ZERO_ROOT,
    CanonicalEventV0,
    hash_bytes,
    hash_canonical_json,
    hash_token_ids,
    make_record,
    verify_records,
)


def _stable_u128(value: str) -> tuple[int, int]:
    digest = hash_bytes(value.encode("utf-8"))
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:16], "little")


@dataclass(slots=True)
class _RequestChain:
    request_id: str
    session_id: str
    model_id: str
    run_id_lo: int
    run_id_hi: int
    path: Path
    sequence_id: int = 0
    logical_token_index: int = 0
    model_step: int = 0
    prev_tip: bytes = ZERO_ROOT
    closed: bool = False
    records: list[dict[str, object]] = field(default_factory=list)


class EvidenceSink:
    """Maintains one logical chain per request, independent of batch slots."""

    def __init__(self, output_dir: Path, *, model_id: str = "unknown") -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self._chains: dict[str, _RequestChain] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _path_for_request(output_dir: Path, request_id: str) -> Path:
        safe_name = uuid.uuid5(uuid.NAMESPACE_URL, request_id).hex
        return output_dir / f"{safe_name}.jsonl"

    def _resume(self, request_id: str) -> Optional[_RequestChain]:
        path = self._path_for_request(self.output_dir, request_id)
        if not path.exists():
            return None
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        verify_records(records)
        first = records[0]
        last = records[-1]
        return _RequestChain(
            request_id=request_id,
            session_id=str(first["session_id"]),
            model_id=self.model_id,
            run_id_lo=int(str(first["run_id_lo"]), 16),
            run_id_hi=int(str(first["run_id_hi"]), 16),
            path=path,
            sequence_id=int(last["sequence_id"]) + 1,
            logical_token_index=sum(int(row["token_count"]) for row in records),
            model_step=max(int(row["model_step"]) for row in records),
            prev_tip=bytes.fromhex(str(last["evidence_root"])),
            closed=int(last["event_type"]) == EVENT_RUN_END,
            records=records,
        )

    def _chain(self, req: Any) -> _RequestChain:
        request_id = str(req.rid)
        chain = self._chains.get(request_id)
        if chain is not None:
            return chain
        chain = self._resume(request_id)
        if chain is not None:
            self._chains[request_id] = chain
            return chain
        session_id = str(getattr(req, "session_id", None) or request_id)
        lo, hi = _stable_u128(f"{session_id}\0{request_id}")
        chain = _RequestChain(
            request_id=request_id,
            session_id=session_id,
            model_id=self.model_id,
            run_id_lo=lo,
            run_id_hi=hi,
            path=self._path_for_request(self.output_dir, request_id),
        )
        self._chains[request_id] = chain
        self._emit(chain, "run_start:sglang_request_v0", EVENT_RUN_START, CADENCE_RUN_BOUNDARY)
        return chain

    def _emit(
        self,
        chain: _RequestChain,
        label: str,
        event_type: int,
        cadence: int,
        *,
        token_start: Optional[int] = None,
        token_ids: Iterable[int] = (),
        sampling_metadata: object = None,
        tool_payload: object = None,
    ) -> dict[str, object]:
        tokens = list(token_ids)
        event = CanonicalEventV0(
            run_id_lo=chain.run_id_lo,
            run_id_hi=chain.run_id_hi,
            sequence_id=chain.sequence_id,
            model_step=chain.model_step,
            token_start=chain.logical_token_index if token_start is None else token_start,
            token_count=len(tokens),
            token_stride=1 if cadence == CADENCE_TOKEN_1 else 0,
            cadence=cadence,
            event_type=event_type,
            model_id_hash=hash_bytes(chain.model_id.encode("utf-8")),
            output_token_hash=hash_token_ids(tokens),
            sampling_metadata_hash=(
                ZERO_ROOT if sampling_metadata is None else hash_canonical_json(sampling_metadata)
            ),
            tool_call_payload_hash=(
                ZERO_ROOT if tool_payload is None else hash_canonical_json(tool_payload)
            ),
            runtime_policy_hash=hash_canonical_json(
                {"runtime": "sglang", "abi": 0, "request_id": chain.request_id}
            ),
        )
        record = make_record(
            event=event,
            label=label,
            prev_tip=chain.prev_tip,
            extensions={
                "session_id": chain.session_id,
                "request_id": chain.request_id,
                "logical_token_index": (
                    chain.logical_token_index if token_start is None else token_start
                ),
            },
        )
        chain.records.append(record)
        chain.prev_tip = bytes.fromhex(str(record["evidence_root"]))
        chain.sequence_id += 1
        return record

    @staticmethod
    def _flush(chain: _RequestChain) -> None:
        temporary = chain.path.with_suffix(".jsonl.tmp")
        payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in chain.records
        )
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(chain.path)

    def record_batch(self, reqs: Iterable[Any], *, batch_kind: str) -> None:
        with self._lock:
            members = [str(req.rid) for req in reqs]
            metadata = {"batch_kind": batch_kind, "members": members}
            for req in reqs:
                chain = self._chain(req)
                self._emit(
                    chain,
                    f"batch:sglang_{batch_kind}_v0",
                    EVENT_DECODE_BATCH,
                    CADENCE_BATCH,
                    sampling_metadata=metadata,
                )
                chain.model_step += 1

    def record_tokens(self, req: Any, token_ids: Iterable[int], *, source: str) -> None:
        with self._lock:
            chain = self._chain(req)
            if chain.closed:
                raise RuntimeError("cannot append tokens to a closed evidence chain")
            for token_id in token_ids:
                self._emit(
                    chain,
                    f"token:sglang_{source}_v0",
                    EVENT_TOKEN_WINDOW,
                    CADENCE_TOKEN_1,
                    token_start=chain.logical_token_index,
                    token_ids=[int(token_id)],
                )
                chain.logical_token_index += 1

    def checkpoint(self, req: Any, *, reason: str, gpu_root: Optional[str] = None) -> None:
        with self._lock:
            chain = self._chain(req)
            self._emit(
                chain,
                f"checkpoint:sglang_{reason}_v0",
                EVENT_CHECKPOINT,
                CADENCE_ADAPTIVE_CHECKPOINT,
                sampling_metadata={"gpu_root": gpu_root, "reason": reason},
            )
            if gpu_root is not None:
                chain.records[-1]["gpu_evidence_root"] = gpu_root
            self._flush(chain)

    def tool_call(self, req: Any, tool_call: Mapping[str, object]) -> None:
        with self._lock:
            chain = self._chain(req)
            self._emit(
                chain, "tool_call:sglang_v0", EVENT_TOOL_CALL, CADENCE_TOOL_CALL,
                tool_payload=dict(tool_call),
            )

    def tool_result(self, req: Any, result: Mapping[str, object]) -> None:
        with self._lock:
            chain = self._chain(req)
            self._emit(
                chain, "tool_result:sglang_v0", EVENT_TOOL_CALL, CADENCE_TOOL_CALL,
                tool_payload={"result": dict(result)},
            )

    def close(self, req: Any, *, reason: str) -> None:
        with self._lock:
            chain = self._chain(req)
            if chain.closed:
                return
            self.checkpoint(req, reason=reason)
            self._emit(
                chain, f"run_end:sglang_{reason}_v0", EVENT_RUN_END, CADENCE_RUN_BOUNDARY,
                sampling_metadata={"reason": reason},
            )
            chain.closed = True
            self._flush(chain)

    def cancel(self, req: Any) -> None:
        self.close(req, reason="cancelled")

    def records_for(self, request_id: str) -> list[dict[str, object]]:
        with self._lock:
            return list(self._chains[request_id].records)

    def append_parsed_tool_calls(
        self, request_id: str, tool_calls: Iterable[Mapping[str, object]]
    ) -> None:
        req = _RequestIdentity(rid=request_id)
        for tool_call in tool_calls:
            self.tool_call(req, tool_call)

    def append_tool_results(
        self, request_id: str, tool_results: Iterable[Mapping[str, object]]
    ) -> None:
        req = _RequestIdentity(rid=request_id)
        for tool_result in tool_results:
            self.tool_result(req, tool_result)

    def finalize_request(self, request_id: str, *, reason: str = "completed") -> None:
        self.close(_RequestIdentity(rid=request_id), reason=reason)


@dataclass(frozen=True, slots=True)
class _RequestIdentity:
    rid: str
    session_id: Optional[str] = None


_configured_output = os.environ.get("SGLANG_EVIDENCE_SIDECAR_DIR")
_configured_model = os.environ.get("SGLANG_EVIDENCE_MODEL_ID", "unknown")
_configured_gpu_library = os.environ.get("SGLANG_EVIDENCE_GPU_LIBRARY")
_sink: Optional[EvidenceSink] = None
_sink_lock = threading.Lock()


def get_evidence_sink() -> Optional[EvidenceSink]:
    global _sink
    if not _configured_output:
        return None
    with _sink_lock:
        if _sink is None:
            _sink = EvidenceSink(Path(_configured_output), model_id=_configured_model)
    return _sink


def evidence_batch(reqs: Iterable[Any], *, batch_kind: str) -> None:
    sink = get_evidence_sink()
    if sink is not None:
        sink.record_batch(reqs, batch_kind=batch_kind)


def evidence_tokens(req: Any, token_ids: Iterable[int], *, source: str) -> None:
    sink = get_evidence_sink()
    if sink is not None:
        sink.record_tokens(req, token_ids, source=source)


def evidence_checkpoint(req: Any, *, reason: str) -> None:
    sink = get_evidence_sink()
    if sink is not None:
        gpu_root = evidence_gpu_checkpoint(req)
        sink.checkpoint(req, reason=reason, gpu_root=gpu_root)


def evidence_finish(req: Any, *, reason: str) -> None:
    sink = get_evidence_sink()
    if sink is not None:
        sink.close(req, reason=reason)


def evidence_cancel(req: Any) -> None:
    sink = get_evidence_sink()
    if sink is not None:
        sink.cancel(req)


class _GpuRuntime:
    def __init__(self, library: Path) -> None:
        from sglang.srt.evidence_sidecar.gpu_provider import GpuEvidenceProvider

        self.provider = GpuEvidenceProvider(library, max_slots=4096, max_updates=2048)
        self.slots: dict[str, int] = {}
        self.free_slots = list(reversed(range(4096)))

    def _slot(self, sink: EvidenceSink, req: Any) -> int:
        request_id = str(req.rid)
        if request_id in self.slots:
            return self.slots[request_id]
        if not self.free_slots:
            raise RuntimeError("GPU evidence request slots exhausted")
        chain = sink._chain(req)
        slot = self.free_slots.pop()
        self.provider.initialize(
            slot,
            run_id_lo=chain.run_id_lo,
            run_id_hi=chain.run_id_hi,
            sequence_id=chain.sequence_id,
            model_step=chain.model_step,
            logical_token_index=chain.logical_token_index,
            model_id_hash=hash_bytes(chain.model_id.encode("utf-8")),
            runtime_policy_hash=hash_canonical_json(
                {"runtime": "sglang", "abi": 0, "request_id": request_id}
            ),
            root=chain.prev_tip,
        )
        self.slots[request_id] = slot
        return slot

    def update(self, sink: EvidenceSink, reqs: list[Any], token_tensor, *, batch_kind: str) -> None:
        members = [str(req.rid) for req in reqs]
        metadata_hash = hash_canonical_json({"batch_kind": batch_kind, "members": members})
        rows = [(self._slot(sink, req), metadata_hash) for req in reqs]
        self.provider.update(rows, token_tensor)

    def checkpoint(self, sink: EvidenceSink, req: Any) -> str:
        request_id = str(req.rid)
        slot = self.slots.get(request_id)
        if slot is None:
            raise RuntimeError("GPU evidence checkpoint has no request slot")
        state = self.provider.checkpoint([slot])[0]
        chain = sink._chain(req)
        expected = (
            chain.sequence_id,
            chain.model_step,
            chain.logical_token_index,
            chain.prev_tip,
        )
        actual = (
            state.sequence_id,
            state.model_step,
            state.logical_token_index,
            state.root,
        )
        if actual != expected:
            raise RuntimeError("GPU evidence checkpoint disagrees with CPU transcript state")
        self.provider.release(slot)
        del self.slots[request_id]
        self.free_slots.append(slot)
        return state.root.hex()


_gpu_runtime: Optional[_GpuRuntime] = None
_gpu_lock = threading.Lock()


def _get_gpu_runtime() -> Optional[_GpuRuntime]:
    global _gpu_runtime
    if not _configured_gpu_library:
        return None
    with _gpu_lock:
        if _gpu_runtime is None:
            _gpu_runtime = _GpuRuntime(Path(_configured_gpu_library))
    return _gpu_runtime


def evidence_gpu_tokens(
    reqs: Iterable[Any], token_tensor, *, batch_kind: str,
    token_indices: Optional[Iterable[int]] = None,
) -> None:
    sink = get_evidence_sink()
    runtime = _get_gpu_runtime()
    values = list(reqs)
    if sink is not None and runtime is not None and values:
        indices = list(token_indices) if token_indices is not None else None
        if indices is not None:
            import torch

            index_tensor = torch.tensor(indices, dtype=torch.int64, device=token_tensor.device)
            token_tensor = token_tensor.index_select(0, index_tensor)
        runtime.update(sink, values, token_tensor, batch_kind=batch_kind)


def evidence_gpu_checkpoint(req: Any) -> Optional[str]:
    sink = get_evidence_sink()
    runtime = _get_gpu_runtime()
    if sink is None or runtime is None:
        return None
    return runtime.checkpoint(sink, req)


def evidence_finalize_response(
    request_id: str,
    *,
    tool_calls: Optional[Iterable[Mapping[str, object]]] = None,
    tool_results: Optional[Iterable[Mapping[str, object]]] = None,
    reason: str = "completed",
) -> None:
    sink = get_evidence_sink()
    if sink is None:
        return
    if tool_results:
        sink.append_tool_results(request_id, tool_results)
    if tool_calls:
        sink.append_parsed_tool_calls(request_id, tool_calls)
    sink.finalize_request(request_id, reason=reason)
