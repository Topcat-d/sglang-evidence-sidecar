"""Portable byte-exact implementation of Smoke Attestation ABI v0."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

ABI_VERSION = 0
LANE_GPU_EVIDENCE_ROOT = 1

CADENCE_RUN_BOUNDARY = 0
CADENCE_BATCH = 1
CADENCE_TOKEN_N = 2
CADENCE_TOKEN_1 = 3
CADENCE_TOOL_CALL = 4
CADENCE_ADAPTIVE_CHECKPOINT = 5

EVENT_RUN_START = 1
EVENT_RUN_END = 2
EVENT_DECODE_BATCH = 3
EVENT_TOKEN_WINDOW = 4
EVENT_TOOL_CALL = 5
EVENT_CHECKPOINT = 6

EVIDENCE_DOMAIN_TILE_SUMMARY = 0x54494C455F524F4F
ZERO_ROOT = bytes(32)
_EVENT_MAGIC = 0x304744454B4F4D53  # SMOKEDG0 as little-endian bytes
_CHAIN_MAGIC = 0x305645454B4F4D53  # SMOKEEV0 as little-endian bytes


def hash_bytes(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def hash_canonical_json(value: object) -> bytes:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hash_bytes(encoded)


def hash_token_ids(token_ids: Iterable[int]) -> bytes:
    values = list(token_ids)
    return hash_bytes(b"".join(struct.pack("<q", token) for token in values))


def stable_run_id(session_id: str, request_id: str) -> tuple[int, int]:
    digest = hash_bytes(f"{session_id}\0{request_id}".encode("utf-8"))
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:16], "little")


@dataclass(frozen=True, slots=True)
class CanonicalEventV0:
    run_id_lo: int
    run_id_hi: int
    sequence_id: int
    model_step: int
    token_start: int
    token_count: int
    token_stride: int
    cadence: int
    event_type: int
    model_id_hash: bytes = ZERO_ROOT
    sampler_config_hash: bytes = ZERO_ROOT
    input_context_tip: bytes = ZERO_ROOT
    output_token_hash: bytes = ZERO_ROOT
    sampling_metadata_hash: bytes = ZERO_ROOT
    tool_call_payload_hash: bytes = ZERO_ROOT
    runtime_policy_hash: bytes = ZERO_ROOT

    def payload(self) -> bytes:
        hashes = (
            self.model_id_hash,
            self.sampler_config_hash,
            self.input_context_tip,
            self.output_token_hash,
            self.sampling_metadata_hash,
            self.tool_call_payload_hash,
            self.runtime_policy_hash,
        )
        if any(len(value) != 32 for value in hashes):
            raise ValueError("canonical event hashes must be 32 bytes")
        payload = struct.pack(
            "<QIIIIQQQQQI",
            _EVENT_MAGIC,
            ABI_VERSION,
            self.cadence,
            self.event_type,
            self.token_stride,
            self.run_id_lo,
            self.run_id_hi,
            self.sequence_id,
            self.model_step,
            self.token_start,
            self.token_count,
        ) + b"".join(hashes)
        if len(payload) != 292:
            raise AssertionError(f"unexpected canonical payload size {len(payload)}")
        return payload

    def digest(self) -> bytes:
        return hash_bytes(self.payload())


def compute_chain_root(
    *, domain: int, sequence_id: int, prev_tip: bytes, digest: bytes
) -> bytes:
    if len(prev_tip) != 32 or len(digest) != 32:
        raise ValueError("chain inputs must be 32 bytes")
    return hash_bytes(
        struct.pack("<QQQ", _CHAIN_MAGIC, domain, sequence_id) + prev_tip + digest
    )


def make_record(
    *,
    event: CanonicalEventV0,
    label: str,
    prev_tip: bytes,
    domain: int = EVIDENCE_DOMAIN_TILE_SUMMARY,
    extensions: Mapping[str, object] | None = None,
) -> dict[str, object]:
    digest = event.digest()
    root = compute_chain_root(
        domain=domain,
        sequence_id=event.sequence_id,
        prev_tip=prev_tip,
        digest=digest,
    )
    record = {
        "schema": "smoke_attestation_transcript_v0",
        "transcript_version": 0,
        "label": label,
        "abi_version": 0,
        "lane": LANE_GPU_EVIDENCE_ROOT,
        "cadence": event.cadence,
        "event_type": event.event_type,
        "flags": 0,
        "token_stride": event.token_stride,
        "run_id_lo": f"{event.run_id_lo:016x}",
        "run_id_hi": f"{event.run_id_hi:016x}",
        "sequence_id": event.sequence_id,
        "model_step": event.model_step,
        "token_start": event.token_start,
        "token_count": event.token_count,
        "payload_bytes": 32,
        "digest": digest.hex(),
        "prev_tip": prev_tip.hex(),
        "domain": f"{domain:016x}",
        "root_status": 0,
        "covered_descriptors": 1,
        "first_sequence_id": event.sequence_id,
        "last_sequence_id": event.sequence_id,
        "evidence_root": root.hex(),
        "model_id_hash": event.model_id_hash.hex(),
        "sampler_config_hash": event.sampler_config_hash.hex(),
        "input_context_tip": event.input_context_tip.hex(),
        "output_token_hash": event.output_token_hash.hex(),
        "sampling_metadata_hash": event.sampling_metadata_hash.hex(),
        "tool_call_payload_hash": event.tool_call_payload_hash.hex(),
        "runtime_policy_hash": event.runtime_policy_hash.hex(),
    }
    if extensions:
        record.update(extensions)
    return record


def verify_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    expected_sequence = 0
    prev_tip = ZERO_ROOT
    run_id = None
    count = 0
    identity = None
    for record in records:
        if record.get("schema") != "smoke_attestation_transcript_v0":
            raise ValueError("unsupported transcript schema")
        sequence_id = int(record["sequence_id"])
        if sequence_id != expected_sequence:
            raise ValueError("non-contiguous sequence id")
        current_run = (
            int(str(record["run_id_lo"]), 16),
            int(str(record["run_id_hi"]), 16),
        )
        if run_id is None:
            run_id = current_run
        elif current_run != run_id:
            raise ValueError("run id changed")
        if bytes.fromhex(str(record["prev_tip"])) != prev_tip:
            raise ValueError("previous tip mismatch")
        event = CanonicalEventV0(
            run_id_lo=current_run[0],
            run_id_hi=current_run[1],
            sequence_id=sequence_id,
            model_step=int(record["model_step"]),
            token_start=int(record["token_start"]),
            token_count=int(record["token_count"]),
            token_stride=int(record["token_stride"]),
            cadence=int(record["cadence"]),
            event_type=int(record["event_type"]),
            model_id_hash=bytes.fromhex(str(record["model_id_hash"])),
            sampler_config_hash=bytes.fromhex(str(record["sampler_config_hash"])),
            input_context_tip=bytes.fromhex(str(record["input_context_tip"])),
            output_token_hash=bytes.fromhex(str(record["output_token_hash"])),
            sampling_metadata_hash=bytes.fromhex(str(record["sampling_metadata_hash"])),
            tool_call_payload_hash=bytes.fromhex(str(record["tool_call_payload_hash"])),
            runtime_policy_hash=bytes.fromhex(str(record["runtime_policy_hash"])),
        )
        digest = bytes.fromhex(str(record["digest"]))
        if event.digest() != digest:
            raise ValueError("canonical event digest mismatch")
        session_id = record.get("session_id")
        request_id = record.get("request_id")
        if session_id is not None or request_id is not None:
            if session_id is None or request_id is None:
                raise ValueError("incomplete transcript identity")
            current_identity = (str(session_id), str(request_id))
            if identity is None:
                identity = current_identity
                if stable_run_id(*identity) != current_run:
                    raise ValueError("run id does not match session/request identity")
            elif current_identity != identity:
                raise ValueError("transcript identity changed")
            if int(record["logical_token_index"]) != int(record["token_start"]):
                raise ValueError("logical token index mismatch")
        expected_root = compute_chain_root(
            domain=int(str(record["domain"]), 16),
            sequence_id=sequence_id,
            prev_tip=prev_tip,
            digest=digest,
        )
        if bytes.fromhex(str(record["evidence_root"])) != expected_root:
            raise ValueError("evidence root mismatch")
        prev_tip = expected_root
        expected_sequence += 1
        count += 1
    if count == 0:
        raise ValueError("empty transcript")
    return {"ok": True, "records": count, "final_root": prev_tip.hex()}


def verify_jsonl(path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    return verify_records(records)
