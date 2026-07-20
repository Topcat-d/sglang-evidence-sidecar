from __future__ import annotations

import sys
import types
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.platform == "win32":
    python_root = Path(__file__).resolve().parents[2] / "python"
    package_paths = {
        "sglang": python_root / "sglang",
        "sglang.srt": python_root / "sglang" / "srt",
        "sglang.srt.evidence_sidecar": python_root
        / "sglang"
        / "srt"
        / "evidence_sidecar",
    }
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules.setdefault(name, module)

from sglang.srt.evidence_sidecar.abi_v0 import verify_jsonl, verify_records
from sglang.srt.evidence_sidecar.gpu_provider import (
    GpuEvidenceProvider,
    RequestStateV0,
    UpdateV0,
)
from sglang.srt.evidence_sidecar.runtime import EvidenceSink, is_evidence_owner


@dataclass
class FakeReq:
    rid: str
    session_id: str | None = None


def test_interleaved_requests_keep_independent_logical_chains(tmp_path):
    sink = EvidenceSink(tmp_path, model_id="tiny-test-model")
    first = FakeReq("request-a", "session")
    second = FakeReq("request-b", "session")

    sink.record_batch([first, second], batch_kind="decode")
    sink.record_tokens(first, [11], source="decode")
    sink.record_tokens(second, [21], source="decode")
    sink.record_batch([second, first], batch_kind="decode")
    sink.record_tokens(second, [22], source="decode")
    sink.record_tokens(first, [12], source="decode")
    sink.close(first, reason="completed")
    sink.close(second, reason="completed")

    first_records = sink.records_for("request-a")
    second_records = sink.records_for("request-b")
    assert verify_records(first_records)["ok"]
    assert verify_records(second_records)["ok"]
    assert [row["token_start"] for row in first_records if row["token_count"]] == [0, 1]
    assert [row["token_start"] for row in second_records if row["token_count"]] == [
        0,
        1,
    ]
    assert first_records[0]["run_id_lo"] != second_records[0]["run_id_lo"]
    assert first_records[0]["request_id"] == "request-a"
    assert first_records[2]["logical_token_index"] == 0
    assert verify_jsonl(next(tmp_path.glob("*.jsonl")))["ok"]


def test_slot_reuse_does_not_reuse_request_identity(tmp_path):
    sink = EvidenceSink(tmp_path)
    old = FakeReq("old-request")
    replacement = FakeReq("replacement-request")
    sink.record_tokens(old, [1], source="prefill")
    sink.close(old, reason="cancelled")
    sink.record_tokens(replacement, [2], source="prefill")

    assert (
        sink.records_for("old-request")[0]["run_id_lo"]
        != sink.records_for("replacement-request")[0]["run_id_lo"]
    )


def test_tool_call_and_result_are_canonicalized(tmp_path):
    sink = EvidenceSink(tmp_path)
    req = FakeReq("tool-request")
    sink.tool_call(req, {"arguments": {"b": 2, "a": 1}, "name": "lookup"})
    first = sink.records_for(req.rid)[-1]["digest"]

    other = FakeReq("tool-request-copy")
    sink.tool_call(other, {"name": "lookup", "arguments": {"a": 1, "b": 2}})
    # Request identity is intentionally part of the digest policy, but canonical
    # JSON ordering makes equivalent payload hashes deterministic within a run.
    assert first != sink.records_for(other.rid)[-1]["digest"]
    sink.tool_result(req, {"ok": True, "value": 3})
    assert sink.records_for(req.rid)[-1]["label"] == "tool_result:sglang_v0"


def test_tamper_is_rejected(tmp_path):
    sink = EvidenceSink(tmp_path)
    req = FakeReq("tamper-request")
    sink.record_tokens(req, [7], source="decode")
    records = sink.records_for(req.rid)
    records[1] = dict(records[1], digest="00" * 32)
    with pytest.raises(ValueError, match="canonical event digest mismatch"):
        verify_records(records)


def test_visible_event_and_identity_tampering_is_rejected(tmp_path):
    sink = EvidenceSink(tmp_path)
    req = FakeReq("visible-tamper-request", "visible-tamper-session")
    sink.record_tokens(req, [7], source="decode")
    records = sink.records_for(req.rid)

    event_tamper = [dict(record) for record in records]
    event_tamper[1]["event_type"] = 99
    with pytest.raises(ValueError, match="canonical event digest mismatch"):
        verify_records(event_tamper)

    identity_tamper = [dict(record) for record in records]
    identity_tamper[0]["session_id"] = "different-session"
    with pytest.raises(ValueError, match="run id does not match"):
        verify_records(identity_tamper)

    index_tamper = [dict(record) for record in records]
    index_tamper[1]["logical_token_index"] = 99
    with pytest.raises(ValueError, match="logical token index mismatch"):
        verify_records(index_tamper)


def test_closed_chain_rejects_late_token(tmp_path):
    sink = EvidenceSink(tmp_path)
    req = FakeReq("closed-request")
    sink.close(req, reason="completed")
    with pytest.raises(RuntimeError, match="closed evidence chain"):
        sink.record_tokens(req, [9], source="decode")


def test_api_process_resumes_chain_and_binds_tool_call(tmp_path):
    scheduler_sink = EvidenceSink(tmp_path, model_id="tiny-test-model")
    req = FakeReq("cross-process-request", "stable-session")
    scheduler_sink.record_batch([req], batch_kind="decode")
    scheduler_sink.record_tokens(req, [41, 42], source="decode")
    scheduler_sink.checkpoint(req, reason="generation_complete")

    api_sink = EvidenceSink(tmp_path, model_id="tiny-test-model")
    api_sink.append_parsed_tool_calls(
        req.rid, [{"name": "lookup", "arguments": {"query": "weather"}}]
    )
    api_sink.append_tool_results(
        req.rid, [{"tool_call_id": "call-1", "content": {"temperature": 72}}]
    )
    api_sink.finalize_request(req.rid, reason="tool_calls")

    records = api_sink.records_for(req.rid)
    assert verify_records(records)["ok"]
    assert records[-2]["label"] == "checkpoint:sglang_tool_calls_v0"
    assert records[-4]["label"] == "tool_call:sglang_v0"
    assert records[-3]["label"] == "tool_result:sglang_v0"
    assert records[-1]["event_type"] == 2


def test_checkpoint_recovery_preserves_sequence_and_token_index(tmp_path):
    req = FakeReq("recover-request", "recover-session")
    before = EvidenceSink(tmp_path)
    before.record_tokens(req, [10, 11], source="decode")
    before.checkpoint(req, reason="periodic")

    after = EvidenceSink(tmp_path)
    after.record_tokens(req, [12], source="decode")
    after.close(req, reason="completed")
    records = after.records_for(req.rid)

    assert verify_records(records)["ok"]
    token_records = [row for row in records if row["token_count"]]
    assert [row["logical_token_index"] for row in token_records] == [0, 1, 2]
    assert [row["sequence_id"] for row in records] == list(range(len(records)))


def test_cancel_is_terminal_and_idempotent(tmp_path):
    req = FakeReq("cancel-request")
    sink = EvidenceSink(tmp_path)
    sink.record_tokens(req, [5], source="decode")
    sink.cancel(req)
    sink.cancel(req)

    records = sink.records_for(req.rid)
    assert verify_records(records)["ok"]
    assert sum(row["event_type"] == 2 for row in records) == 1
    assert records[-1]["label"] == "run_end:sglang_cancelled_v0"


def test_only_attention_parallel_leader_owns_evidence():
    leader = SimpleNamespace(attn_tp_rank=0, attn_cp_rank=0)
    tp_replica = SimpleNamespace(attn_tp_rank=1, attn_cp_rank=0)
    cp_replica = SimpleNamespace(attn_tp_rank=0, attn_cp_rank=1)

    assert is_evidence_owner(leader)
    assert not is_evidence_owner(tp_replica)
    assert not is_evidence_owner(cp_replica)


def test_gpu_ctypes_layout_matches_c_api():
    assert ctypes.sizeof(RequestStateV0) == 136
    assert ctypes.sizeof(UpdateV0) == 40


@pytest.mark.skipif(
    not os.environ.get("SGLANG_EVIDENCE_TEST_LIBRARY"),
    reason="compiled CUDA evidence library not provided",
)
def test_gpu_library_lifecycle():
    provider = GpuEvidenceProvider(
        Path(os.environ["SGLANG_EVIDENCE_TEST_LIBRARY"]), max_slots=2, max_updates=2
    )
    provider.initialize(
        0,
        run_id_lo=1,
        run_id_hi=2,
        sequence_id=3,
        model_step=4,
        logical_token_index=5,
        model_id_hash=bytes(range(32)),
        runtime_policy_hash=bytes(range(32, 64)),
        root=bytes(range(64, 96)),
    )
    state = provider.checkpoint([0])[0]
    assert (state.sequence_id, state.model_step, state.logical_token_index) == (3, 4, 5)
    assert state.root == bytes(range(64, 96))
    provider.release(0)
    provider.close()
