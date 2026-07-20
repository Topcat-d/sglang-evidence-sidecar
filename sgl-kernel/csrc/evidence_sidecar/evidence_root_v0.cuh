#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace sglang_evidence {

constexpr uint64_t kEventMagic = 0x304744454b4f4d53ull;
constexpr uint64_t kChainMagic = 0x305645454b4f4d53ull;
constexpr uint64_t kEvidenceDomain = 0x54494c455f524f4full;

struct alignas(16) RequestStateV0 {
    uint64_t run_id_lo;
    uint64_t run_id_hi;
    uint64_t sequence_id;
    uint64_t model_step;
    uint64_t logical_token_index;
    uint8_t model_id_hash[32];
    uint8_t runtime_policy_hash[32];
    uint8_t root[32];
};

struct UpdateV0 {
    uint32_t slot;
    uint32_t reserved;
    uint8_t batch_metadata_hash[32];
};

__device__ __constant__ uint32_t kSha256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t load_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

__device__ __forceinline__ void store_be32(uint8_t *p, uint32_t x) {
    p[0] = (uint8_t)(x >> 24);
    p[1] = (uint8_t)(x >> 16);
    p[2] = (uint8_t)(x >> 8);
    p[3] = (uint8_t)x;
}

__device__ __forceinline__ void store_le32(uint8_t *p, uint32_t x) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(x >> (8 * i));
}

__device__ __forceinline__ void store_le64(uint8_t *p, uint64_t x) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(x >> (8 * i));
}

__device__ __forceinline__ void compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = load_be32(block + 4 * i);
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + kSha256[i] + w[i];
        uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ __forceinline__ void sha256(const uint8_t *data, int len, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    int offset = 0;
    while (len - offset >= 64) {
        compress(state, data + offset);
        offset += 64;
    }
    uint8_t tail[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) tail[i] = 0;
    int remaining = len - offset;
    for (int i = 0; i < remaining; ++i) tail[i] = data[offset + i];
    tail[remaining] = 0x80;
    int tail_bytes = remaining < 56 ? 64 : 128;
    uint64_t bit_len = (uint64_t)len * 8ull;
    for (int i = 0; i < 8; ++i) tail[tail_bytes - 1 - i] = (uint8_t)(bit_len >> (8 * i));
    compress(state, tail);
    if (tail_bytes == 128) compress(state, tail + 64);
    #pragma unroll
    for (int i = 0; i < 8; ++i) store_be32(out + 4 * i, state[i]);
}

__device__ __forceinline__ void canonical_digest(
    const RequestStateV0 &state, uint32_t cadence, uint32_t event_type,
    uint32_t token_stride, uint32_t token_count, const uint8_t output_hash[32],
    const uint8_t metadata_hash[32], uint8_t digest[32]) {
    uint8_t payload[292];
    #pragma unroll
    for (int i = 0; i < 292; ++i) payload[i] = 0;
    store_le64(payload + 0, kEventMagic);
    store_le32(payload + 8, 0);
    store_le32(payload + 12, cadence);
    store_le32(payload + 16, event_type);
    store_le32(payload + 20, token_stride);
    store_le64(payload + 24, state.run_id_lo);
    store_le64(payload + 32, state.run_id_hi);
    store_le64(payload + 40, state.sequence_id);
    store_le64(payload + 48, state.model_step);
    store_le64(payload + 56, state.logical_token_index);
    store_le32(payload + 64, token_count);
    for (int i = 0; i < 32; ++i) {
        payload[68 + i] = state.model_id_hash[i];
        payload[164 + i] = output_hash[i];
        payload[196 + i] = metadata_hash[i];
        payload[260 + i] = state.runtime_policy_hash[i];
    }
    sha256(payload, 292, digest);
}

__device__ __forceinline__ void advance_root(RequestStateV0 &state, const uint8_t digest[32]) {
    uint8_t message[88];
    store_le64(message + 0, kChainMagic);
    store_le64(message + 8, kEvidenceDomain);
    store_le64(message + 16, state.sequence_id);
    for (int i = 0; i < 32; ++i) {
        message[24 + i] = state.root[i];
        message[56 + i] = digest[i];
    }
    sha256(message, 88, state.root);
    ++state.sequence_id;
}

__global__ void update_token_roots_v0(
    RequestStateV0 *states, const UpdateV0 *updates, const int64_t *token_ids,
    uint32_t update_count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= update_count) return;
    RequestStateV0 &state = states[updates[index].slot];
    uint8_t empty_hash[32], output_hash[32], zero[32], digest[32], token_bytes[8];
    for (int i = 0; i < 32; ++i) zero[i] = 0;
    sha256(nullptr, 0, empty_hash);

    canonical_digest(state, 1, 3, 0, 0, empty_hash,
                     updates[index].batch_metadata_hash, digest);
    advance_root(state, digest);

    store_le64(token_bytes, (uint64_t)token_ids[index]);
    sha256(token_bytes, 8, output_hash);
    canonical_digest(state, 3, 4, 1, 1, output_hash, zero, digest);
    advance_root(state, digest);
    ++state.logical_token_index;
    ++state.model_step;
}

}  // namespace sglang_evidence
