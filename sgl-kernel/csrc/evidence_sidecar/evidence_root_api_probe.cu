#include "evidence_root_api.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { int status_ = (call); if (status_ != 0) { \
    std::fprintf(stderr, "api_error,%s,%d,%d\n", __FILE__, __LINE__, status_); return 1; \
} } while (0)
#define CUDA_CHECK(call) do { cudaError_t status_ = (call); if (status_ != cudaSuccess) { \
    std::fprintf(stderr, "cuda_error,%s,%d,%s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); return 1; \
} } while (0)

static void print_hex(const uint8_t value[32]) {
    for (int i = 0; i < 32; ++i) std::printf("%02x", value[i]);
}

int main(int argc, char **argv) {
    constexpr uint32_t requests = 4;
    uint32_t steps = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 16;
    if (steps == 0) return 2;
    SglangEvidenceContext *context = nullptr;
    CHECK(sglang_evidence_create_v0(requests, requests, &context));
    std::vector<SglangEvidenceUpdateV0> updates(requests);
    for (uint32_t i = 0; i < requests; ++i) {
        SglangEvidenceRequestStateV0 state = {};
        state.run_id_lo = 0x1000ull + i;
        state.run_id_hi = 0x2000ull + i;
        for (int j = 0; j < 32; ++j) {
            state.model_id_hash[j] = (uint8_t)(j + i);
            state.runtime_policy_hash[j] = (uint8_t)(0xa0 + j + i);
            updates[i].batch_metadata_hash[j] = (uint8_t)(0x40 + j);
        }
        updates[i].slot = i;
        CHECK(sglang_evidence_initialize_slot_v0(context, i, &state));
    }

    cudaStream_t producer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    int64_t *device_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&device_tokens, requests * sizeof(*device_tokens)));
    std::vector<int64_t> tokens(requests);
    for (uint32_t step = 0; step < steps; ++step) {
        for (uint32_t i = 0; i < requests; ++i) {
            tokens[i] = 100000 + step * requests + i;
            for (int j = 0; j < 32; ++j) {
                updates[i].batch_metadata_hash[j] = (uint8_t)(0x40 + j + step);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(device_tokens, tokens.data(), requests * sizeof(*device_tokens),
                                   cudaMemcpyHostToDevice, producer));
        CHECK(sglang_evidence_update_tokens_v0(
            context, updates.data(), device_tokens, requests, reinterpret_cast<void *>(producer)));
    }

    uint32_t slots[requests] = {0, 1, 2, 3};
    SglangEvidenceRequestStateV0 states[requests] = {};
    CHECK(sglang_evidence_checkpoint_v0(context, slots, requests, states));
    for (uint32_t i = 0; i < requests; ++i) {
        std::printf("api_root,%u,%llu,%llu,", i,
                    (unsigned long long)states[i].sequence_id,
                    (unsigned long long)states[i].logical_token_index);
        print_hex(states[i].root);
        std::printf("\n");
    }
    std::printf("api_probe,producer_stream_wait,1\n");
    std::printf("api_probe,checkpoint_only_export,1\n");
    std::printf("api_probe,steps,%u\n", steps);

    cudaFree(device_tokens);
    cudaStreamDestroy(producer);
    sglang_evidence_destroy_v0(context);
    return 0;
}
