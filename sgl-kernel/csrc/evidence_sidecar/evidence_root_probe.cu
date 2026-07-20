#include "evidence_root_v0.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using sglang_evidence::RequestStateV0;
using sglang_evidence::UpdateV0;

#define CUDA_OK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        std::fprintf(stderr, "cuda_error,%s,%d,%s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        return 1; \
    } \
} while (0)

static void print_hex(const uint8_t *value, size_t size) {
    for (size_t i = 0; i < size; ++i) std::printf("%02x", value[i]);
}

int main(int argc, char **argv) {
    int requests = argc > 1 ? std::atoi(argv[1]) : 4;
    int steps = argc > 2 ? std::atoi(argv[2]) : 16;
    int repeats = argc > 3 ? std::atoi(argv[3]) : 20;
    if (requests <= 0 || steps <= 0 || repeats <= 0) return 2;

    std::vector<RequestStateV0> initial(requests);
    std::vector<UpdateV0> updates(requests);
    std::vector<int64_t> tokens((size_t)requests * steps);
    for (int i = 0; i < requests; ++i) {
        initial[i].run_id_lo = 0x1000ull + (uint64_t)i;
        initial[i].run_id_hi = 0x2000ull + (uint64_t)i;
        for (int j = 0; j < 32; ++j) {
            initial[i].model_id_hash[j] = (uint8_t)(j + i);
            initial[i].runtime_policy_hash[j] = (uint8_t)(0xa0 + j + i);
            updates[i].batch_metadata_hash[j] = (uint8_t)(0x40 + j);
        }
        updates[i].slot = (uint32_t)i;
    }

    RequestStateV0 *d_states = nullptr;
    UpdateV0 *d_updates = nullptr;
    int64_t *d_tokens = nullptr;
    CUDA_OK(cudaMalloc(&d_states, initial.size() * sizeof(RequestStateV0)));
    CUDA_OK(cudaMalloc(&d_updates, updates.size() * sizeof(UpdateV0)));
    CUDA_OK(cudaMalloc(&d_tokens, tokens.size() * sizeof(int64_t)));
    cudaStream_t stream;
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));

    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < requests; ++i) {
            tokens[(size_t)step * requests + i] = 100000 + step * requests + i;
        }
    }
    std::vector<float> samples;
    samples.reserve(repeats);
    std::vector<RequestStateV0> result(requests);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        CUDA_OK(cudaMemcpyAsync(d_states, initial.data(), initial.size() * sizeof(RequestStateV0),
                                cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_updates, updates.data(), updates.size() * sizeof(UpdateV0),
                                cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_tokens, tokens.data(), tokens.size() * sizeof(int64_t),
                                cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaEventRecord(start, stream));
        for (int step = 0; step < steps; ++step) {
            sglang_evidence::update_token_roots_v0<<<(requests + 127) / 128, 128, 0, stream>>>(
                d_states, d_updates, d_tokens + (size_t)step * requests, (uint32_t)requests);
            CUDA_OK(cudaGetLastError());
        }
        CUDA_OK(cudaEventRecord(stop, stream));
        CUDA_OK(cudaMemcpyAsync(result.data(), d_states, result.size() * sizeof(RequestStateV0),
                                cudaMemcpyDeviceToHost, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        float elapsed = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(elapsed);
    }

    std::sort(samples.begin(), samples.end());
    float median_ms = samples[samples.size() / 2];
    size_t p95_index = (samples.size() * 95 + 99) / 100 - 1;
    float p95_ms = samples[p95_index];

    std::printf("probe,requests,%d\nprobe,steps,%d\nprobe,repeats,%d\n", requests, steps, repeats);
    std::printf("probe,median_device_ms,%.9f\n", median_ms);
    std::printf("probe,p95_device_ms,%.9f\n", p95_ms);
    std::printf("probe,checkpoint_copies_per_repeat,1\n");
    std::printf("probe,auxiliary_stream,1\n");
    for (int i = 0; i < requests; ++i) {
        std::printf("root,%d,%llu,%llu,", i,
                    (unsigned long long)result[i].sequence_id,
                    (unsigned long long)result[i].logical_token_index);
        print_hex(result[i].root, 32);
        std::printf("\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_tokens);
    cudaFree(d_updates);
    cudaFree(d_states);
    return 0;
}
