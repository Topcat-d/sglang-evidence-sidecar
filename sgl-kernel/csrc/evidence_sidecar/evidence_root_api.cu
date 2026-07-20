#include "evidence_root_api.h"
#include "evidence_root_v0.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <new>

constexpr uint32_t kUpdateStagingDepth = 256;

static_assert(sizeof(SglangEvidenceRequestStateV0) == sizeof(sglang_evidence::RequestStateV0));
static_assert(sizeof(SglangEvidenceUpdateV0) == sizeof(sglang_evidence::UpdateV0));

struct SglangEvidenceContext {
    uint32_t max_slots = 0;
    uint32_t max_updates = 0;
    sglang_evidence::RequestStateV0 *states = nullptr;
    sglang_evidence::UpdateV0 *updates = nullptr;
    int64_t *token_staging = nullptr;
    sglang_evidence::UpdateV0 *updates_staging = nullptr;
    cudaEvent_t *updates_staged = nullptr;
    uint8_t *updates_staging_used = nullptr;
    uint32_t updates_staging_cursor = 0;
    SglangEvidenceRequestStateV0 *checkpoint_staging = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t producer_ready = nullptr;
    cudaEvent_t tokens_staged = nullptr;
};

static int as_status(cudaError_t error) {
    return error == cudaSuccess ? 0 : (int)error;
}

extern "C" int sglang_evidence_create_v0(
    uint32_t max_slots, uint32_t max_updates, SglangEvidenceContext **context) {
    if (!context || max_slots == 0 || max_updates == 0) return (int)cudaErrorInvalidValue;
    *context = nullptr;
    SglangEvidenceContext *value = new (std::nothrow) SglangEvidenceContext();
    if (!value) return (int)cudaErrorMemoryAllocation;
    value->max_slots = max_slots;
    value->max_updates = max_updates;
    cudaError_t error = cudaStreamCreateWithFlags(&value->stream, cudaStreamNonBlocking);
    if (error == cudaSuccess) error = cudaEventCreateWithFlags(&value->producer_ready, cudaEventDisableTiming);
    if (error == cudaSuccess) error = cudaEventCreateWithFlags(&value->tokens_staged, cudaEventDisableTiming);
    if (error == cudaSuccess) error = cudaMalloc(&value->states, max_slots * sizeof(*value->states));
    if (error == cudaSuccess) error = cudaMemsetAsync(value->states, 0, max_slots * sizeof(*value->states), value->stream);
    if (error == cudaSuccess) error = cudaMalloc(&value->updates, max_updates * sizeof(*value->updates));
    if (error == cudaSuccess) error = cudaMalloc(&value->token_staging, max_updates * sizeof(*value->token_staging));
    if (error == cudaSuccess) {
        error = cudaMallocHost(
            &value->updates_staging,
            (size_t)max_updates * kUpdateStagingDepth * sizeof(*value->updates_staging));
    }
    if (error == cudaSuccess) {
        value->updates_staged = new (std::nothrow) cudaEvent_t[kUpdateStagingDepth]();
        value->updates_staging_used = new (std::nothrow) uint8_t[kUpdateStagingDepth]();
        if (!value->updates_staged || !value->updates_staging_used) {
            error = cudaErrorMemoryAllocation;
        }
    }
    for (uint32_t i = 0; i < kUpdateStagingDepth && error == cudaSuccess; ++i) {
        error = cudaEventCreateWithFlags(&value->updates_staged[i], cudaEventDisableTiming);
    }
    if (error == cudaSuccess) error = cudaMallocHost(&value->checkpoint_staging, max_slots * sizeof(*value->checkpoint_staging));
    if (error == cudaSuccess) error = cudaStreamSynchronize(value->stream);
    if (error != cudaSuccess) {
        sglang_evidence_destroy_v0(value);
        return (int)error;
    }
    *context = value;
    return 0;
}

extern "C" int sglang_evidence_initialize_slot_v0(
    SglangEvidenceContext *context, uint32_t slot,
    const SglangEvidenceRequestStateV0 *state) {
    if (!context || !state || slot >= context->max_slots) return (int)cudaErrorInvalidValue;
    cudaError_t error = cudaMemcpyAsync(
        context->states + slot, state, sizeof(*state), cudaMemcpyHostToDevice, context->stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->stream);
    return as_status(error);
}

extern "C" int sglang_evidence_update_tokens_v0(
    SglangEvidenceContext *context, const SglangEvidenceUpdateV0 *updates,
    const int64_t *device_token_ids, uint32_t update_count, void *producer_stream) {
    if (!context || !updates || !device_token_ids || update_count == 0 ||
        update_count > context->max_updates) return (int)cudaErrorInvalidValue;
    for (uint32_t i = 0; i < update_count; ++i) {
        if (updates[i].slot >= context->max_slots) return (int)cudaErrorInvalidValue;
    }
    uint32_t staging_index = context->updates_staging_cursor++ % kUpdateStagingDepth;
    if (context->updates_staging_used[staging_index]) {
        cudaError_t staging_error = cudaEventSynchronize(context->updates_staged[staging_index]);
        if (staging_error != cudaSuccess) return (int)staging_error;
    }
    sglang_evidence::UpdateV0 *staging =
        context->updates_staging + (size_t)staging_index * context->max_updates;
    std::memcpy(staging, updates, update_count * sizeof(*updates));
    cudaStream_t producer = reinterpret_cast<cudaStream_t>(producer_stream);
    cudaError_t error = cudaEventRecord(context->producer_ready, producer);
    if (error == cudaSuccess) error = cudaStreamWaitEvent(context->stream, context->producer_ready, 0);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(context->updates, staging,
                                update_count * sizeof(*context->updates),
                                cudaMemcpyHostToDevice, context->stream);
    }
    if (error == cudaSuccess) {
        error = cudaEventRecord(context->updates_staged[staging_index], context->stream);
        context->updates_staging_used[staging_index] = 1;
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(context->token_staging, device_token_ids,
                                update_count * sizeof(*context->token_staging),
                                cudaMemcpyDeviceToDevice, context->stream);
    }
    if (error == cudaSuccess) error = cudaEventRecord(context->tokens_staged, context->stream);
    if (error == cudaSuccess) error = cudaStreamWaitEvent(producer, context->tokens_staged, 0);
    if (error == cudaSuccess) {
        sglang_evidence::update_token_roots_v0<<<(update_count + 127) / 128, 128, 0, context->stream>>>(
            context->states, context->updates, context->token_staging, update_count);
        error = cudaGetLastError();
    }
    return as_status(error);
}

extern "C" int sglang_evidence_checkpoint_v0(
    SglangEvidenceContext *context, const uint32_t *slots, uint32_t slot_count,
    SglangEvidenceRequestStateV0 *states_out) {
    if (!context || !slots || !states_out || slot_count == 0 ||
        slot_count > context->max_slots) return (int)cudaErrorInvalidValue;
    cudaError_t error = cudaSuccess;
    for (uint32_t i = 0; i < slot_count && error == cudaSuccess; ++i) {
        if (slots[i] >= context->max_slots) return (int)cudaErrorInvalidValue;
        error = cudaMemcpyAsync(context->checkpoint_staging + i, context->states + slots[i],
                                sizeof(*states_out), cudaMemcpyDeviceToHost, context->stream);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(context->stream);
    if (error == cudaSuccess) {
        std::memcpy(states_out, context->checkpoint_staging, slot_count * sizeof(*states_out));
    }
    return as_status(error);
}

extern "C" int sglang_evidence_release_slot_v0(SglangEvidenceContext *context, uint32_t slot) {
    if (!context || slot >= context->max_slots) return (int)cudaErrorInvalidValue;
    return as_status(cudaMemsetAsync(
        context->states + slot, 0, sizeof(*context->states), context->stream));
}

extern "C" void *sglang_evidence_auxiliary_stream_v0(SglangEvidenceContext *context) {
    return context ? reinterpret_cast<void *>(context->stream) : nullptr;
}

extern "C" void sglang_evidence_destroy_v0(SglangEvidenceContext *context) {
    if (!context) return;
    if (context->stream) cudaStreamSynchronize(context->stream);
    if (context->checkpoint_staging) cudaFreeHost(context->checkpoint_staging);
    if (context->updates_staging) cudaFreeHost(context->updates_staging);
    if (context->updates_staged) {
        for (uint32_t i = 0; i < kUpdateStagingDepth; ++i) {
            if (context->updates_staged[i]) cudaEventDestroy(context->updates_staged[i]);
        }
    }
    delete[] context->updates_staged;
    delete[] context->updates_staging_used;
    if (context->updates) cudaFree(context->updates);
    if (context->token_staging) cudaFree(context->token_staging);
    if (context->states) cudaFree(context->states);
    if (context->producer_ready) cudaEventDestroy(context->producer_ready);
    if (context->tokens_staged) cudaEventDestroy(context->tokens_staged);
    if (context->stream) cudaStreamDestroy(context->stream);
    delete context;
}
