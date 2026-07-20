#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define SGLANG_EVIDENCE_API __declspec(dllexport)
#else
#define SGLANG_EVIDENCE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SglangEvidenceContext SglangEvidenceContext;

typedef struct SglangEvidenceRequestStateV0 {
    uint64_t run_id_lo;
    uint64_t run_id_hi;
    uint64_t sequence_id;
    uint64_t model_step;
    uint64_t logical_token_index;
    uint8_t model_id_hash[32];
    uint8_t runtime_policy_hash[32];
    uint8_t root[32];
} SglangEvidenceRequestStateV0;

typedef struct SglangEvidenceUpdateV0 {
    uint32_t slot;
    uint32_t reserved;
    uint8_t batch_metadata_hash[32];
} SglangEvidenceUpdateV0;

SGLANG_EVIDENCE_API int sglang_evidence_create_v0(
    uint32_t max_slots,
    uint32_t max_updates,
    SglangEvidenceContext **context);

SGLANG_EVIDENCE_API int sglang_evidence_initialize_slot_v0(
    SglangEvidenceContext *context,
    uint32_t slot,
    const SglangEvidenceRequestStateV0 *state);

SGLANG_EVIDENCE_API int sglang_evidence_update_tokens_v0(
    SglangEvidenceContext *context,
    const SglangEvidenceUpdateV0 *updates,
    const int64_t *device_token_ids,
    uint32_t update_count,
    void *producer_stream);

SGLANG_EVIDENCE_API int sglang_evidence_checkpoint_v0(
    SglangEvidenceContext *context,
    const uint32_t *slots,
    uint32_t slot_count,
    SglangEvidenceRequestStateV0 *states_out);

SGLANG_EVIDENCE_API int sglang_evidence_release_slot_v0(
    SglangEvidenceContext *context,
    uint32_t slot);

SGLANG_EVIDENCE_API void *sglang_evidence_auxiliary_stream_v0(
    SglangEvidenceContext *context);

SGLANG_EVIDENCE_API void sglang_evidence_destroy_v0(
    SglangEvidenceContext *context);

#ifdef __cplusplus
}
#endif
