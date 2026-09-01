#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "backends/hip/hip_fused_kernels.h"
#include <string>

#include <cmath>

namespace hybridai::hip_fused {

namespace {

constexpr int kBlockSize = 256;

template <typename T>
__device__ float load_value(const T* data, int64_t index) {
    return static_cast<float>(data[index]);
}

template <typename T>
__device__ void store_value(T* data, int64_t index, float value) {
    data[index] = static_cast<T>(value);
    return;
}

template <typename T>
__global__ void qk_rmsnorm_rope_kernel(
    T* query_dst, const T* query_src, const T* query_weight,
    T* key_dst, const T* key_src, const T* key_weight, int64_t seq_len,
    int64_t num_query_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t rope_head_dim, int64_t position_offset, float rope_base,
    float eps, bool add_unit_offset) {
    const int64_t logical_head = static_cast<int64_t>(blockIdx.x);
    const int64_t total_heads = seq_len * (num_query_heads + num_kv_heads);
    if (logical_head >= total_heads) return;

    const bool is_query = logical_head < seq_len * num_query_heads;
    const int64_t head_index = is_query
        ? logical_head
        : logical_head - seq_len * num_query_heads;
    const int64_t heads = is_query ? num_query_heads : num_kv_heads;
    const int64_t token = head_index / heads;
    const int64_t head = head_index - token * heads;
    const int64_t base_index = head_index * head_dim;
    const T* src = is_query ? query_src : key_src;
    const T* weight = is_query ? query_weight : key_weight;
    T* dst = is_query ? query_dst : key_dst;

    __shared__ float reduction[kBlockSize];
    float sum = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float value = load_value(src, base_index + d);
        sum += value * value;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduction[0] / static_cast<float>(head_dim) + eps);
    const int64_t half = rope_head_dim / 2;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = load_value(src, base_index + d) * inv_rms *
            (load_value(weight, d) + (add_unit_offset ? 1.0f : 0.0f));
        if (d < rope_head_dim) {
            const int64_t pair = d < half ? d : d - half;
            const float theta = static_cast<float>(position_offset + token) /
                powf(rope_base, 2.0f * static_cast<float>(pair) /
                                  static_cast<float>(rope_head_dim));
            const float cosine = cosf(theta);
            const float sine = sinf(theta);
            const float x = load_value(src, base_index + pair) * inv_rms *
                (load_value(weight, pair) + (add_unit_offset ? 1.0f : 0.0f));
            const float y = load_value(src, base_index + pair + half) * inv_rms *
                (load_value(weight, pair + half) +
                 (add_unit_offset ? 1.0f : 0.0f));
            value = d < half ? x * cosine - y * sine
                             : y * cosine + x * sine;
        }
        store_value(dst, base_index + d, value);
    }
}

template <typename T>
__global__ void q_gate_rmsnorm_rope_kernel(
    T* query_dst, T* gate_dst, const T* source, const T* query_weight,
    int64_t seq_len, int64_t num_query_heads, int64_t head_dim,
    int64_t rope_head_dim, int64_t position_offset, float rope_base,
    float eps, bool add_unit_offset) {
    const int64_t head_index = static_cast<int64_t>(blockIdx.x);
    const int64_t total_heads = seq_len * num_query_heads;
    if (head_index >= total_heads) return;
    const int64_t source_base = head_index * 2 * head_dim;
    const int64_t output_base = head_index * head_dim;

    __shared__ float reduction[kBlockSize];
    float sum = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float value = load_value(source, source_base + d);
        sum += value * value;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduction[0] / static_cast<float>(head_dim) + eps);
    const int64_t half = rope_head_dim / 2;
    const int64_t token = head_index / num_query_heads;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float scale = load_value(query_weight, d) +
            (add_unit_offset ? 1.0f : 0.0f);
        const float q = load_value(source, source_base + d) * inv_rms * scale;
        const float gate = load_value(source, source_base + head_dim + d);
        store_value(gate_dst, output_base + d, gate);
        float value = q;
        if (d < rope_head_dim) {
            const int64_t pair = d < half ? d : d - half;
            const float theta = static_cast<float>(position_offset + token) /
                powf(rope_base, 2.0f * static_cast<float>(pair) /
                                  static_cast<float>(rope_head_dim));
            const float cosine = cosf(theta);
            const float sine = sinf(theta);
            const int64_t pair2 = pair + half;
            const float x = load_value(source, source_base + pair) * inv_rms *
                (load_value(query_weight, pair) +
                 (add_unit_offset ? 1.0f : 0.0f));
            const float y = load_value(source, source_base + pair2) * inv_rms *
                (load_value(query_weight, pair2) +
                 (add_unit_offset ? 1.0f : 0.0f));
            value = d < half ? x * cosine - y * sine
                             : y * cosine + x * sine;
        }
        store_value(query_dst, output_base + d, value);
    }
}

template <typename T>
Status launch_qk_rmsnorm_rope(
    void* query_dst, const void* query_src, const void* query_weight,
    void* key_dst, const void* key_src, const void* key_weight,
    int64_t seq_len, int64_t num_query_heads, int64_t num_kv_heads,
    int64_t head_dim, int64_t rope_head_dim, int64_t position_offset,
    float rope_base, float eps, bool add_unit_offset, void* stream_handle) {
    const int64_t heads = seq_len * (num_query_heads + num_kv_heads);
    hipStream_t stream = static_cast<hipStream_t>(stream_handle);
    hipLaunchKernelGGL(
        qk_rmsnorm_rope_kernel<T>, dim3(static_cast<unsigned int>(heads)),
        dim3(kBlockSize), 0, stream, static_cast<T*>(query_dst),
        static_cast<const T*>(query_src), static_cast<const T*>(query_weight),
        static_cast<T*>(key_dst), static_cast<const T*>(key_src),
        static_cast<const T*>(key_weight), seq_len, num_query_heads,
        num_kv_heads, head_dim, rope_head_dim, position_offset, rope_base, eps,
        add_unit_offset);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        return Status(StatusCode::BackendError,
                      std::string("launch fused QK RMSNorm/RoPE: ") +
                          hipGetErrorString(error));
    }
    return Status::OK();
}

template <typename T>
Status launch_q_gate_rmsnorm_rope(
    void* query_dst, void* gate_dst, const void* source,
    const void* query_weight, int64_t seq_len, int64_t num_query_heads,
    int64_t head_dim, int64_t rope_head_dim, int64_t position_offset,
    float rope_base, float eps, bool add_unit_offset, void* stream_handle) {
    const int64_t heads = seq_len * num_query_heads;
    hipStream_t stream = static_cast<hipStream_t>(stream_handle);
    hipLaunchKernelGGL(
        q_gate_rmsnorm_rope_kernel<T>, dim3(static_cast<unsigned int>(heads)),
        dim3(kBlockSize), 0, stream, static_cast<T*>(query_dst),
        static_cast<T*>(gate_dst), static_cast<const T*>(source),
        static_cast<const T*>(query_weight), seq_len, num_query_heads,
        head_dim, rope_head_dim, position_offset, rope_base, eps,
        add_unit_offset);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        return Status(StatusCode::BackendError,
                      std::string("launch fused Q/gate RMSNorm/RoPE: ") +
                          hipGetErrorString(error));
    }
    return Status::OK();
}

} // namespace
Status qk_rmsnorm_rope(
    void* query_dst, const void* query_src, const void* query_weight,
    void* key_dst, const void* key_src, const void* key_weight, DType dtype,
    int64_t seq_len, int64_t num_query_heads, int64_t num_kv_heads,
    int64_t head_dim, int64_t rope_head_dim, int64_t position_offset,
    float rope_base, float eps, bool add_unit_offset, void* stream_handle) {
    if (query_dst == nullptr || query_src == nullptr || query_weight == nullptr ||
        key_dst == nullptr || key_src == nullptr || key_weight == nullptr ||
        seq_len <= 0 || num_query_heads <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || rope_head_dim <= 0 || rope_head_dim > head_dim ||
        (rope_head_dim & 1) != 0) {
        return Status(StatusCode::InvalidArgument,
                      "invalid fused QK RMSNorm/RoPE arguments");
    }
    if (dtype == DType::FP32) {
        return launch_qk_rmsnorm_rope<float>(
            query_dst, query_src, query_weight, key_dst, key_src, key_weight,
            seq_len, num_query_heads, num_kv_heads, head_dim, rope_head_dim,
            position_offset, rope_base, eps, add_unit_offset, stream_handle);
    }
    if (dtype == DType::BF16) {
        return launch_qk_rmsnorm_rope<hip_bfloat16>(
            query_dst, query_src, query_weight, key_dst, key_src, key_weight,
            seq_len, num_query_heads, num_kv_heads, head_dim, rope_head_dim,
            position_offset, rope_base, eps, add_unit_offset, stream_handle);
    }
    return Status(StatusCode::UnsupportedDType,
                  "fused QK RMSNorm/RoPE supports FP32 and BF16");
}

Status q_gate_rmsnorm_rope(
    void* query_dst, void* gate_dst, const void* source,
    const void* query_weight, DType dtype, int64_t seq_len,
    int64_t num_query_heads, int64_t head_dim, int64_t rope_head_dim,
    int64_t position_offset, float rope_base, float eps,
    bool add_unit_offset, void* stream_handle) {
    if (query_dst == nullptr || gate_dst == nullptr || source == nullptr ||
        query_weight == nullptr || seq_len <= 0 || num_query_heads <= 0 ||
        head_dim <= 0 || rope_head_dim <= 0 || rope_head_dim > head_dim ||
        (rope_head_dim & 1) != 0) {
        return Status(StatusCode::InvalidArgument,
                      "invalid fused Q/gate RMSNorm/RoPE arguments");
    }
    if (dtype == DType::FP32) {
        return launch_q_gate_rmsnorm_rope<float>(
            query_dst, gate_dst, source, query_weight, seq_len,
            num_query_heads, head_dim, rope_head_dim, position_offset,
            rope_base, eps, add_unit_offset, stream_handle);
    }
    if (dtype == DType::BF16) {
        return launch_q_gate_rmsnorm_rope<hip_bfloat16>(
            query_dst, gate_dst, source, query_weight, seq_len,
            num_query_heads, head_dim, rope_head_dim, position_offset,
            rope_base, eps, add_unit_offset, stream_handle);
    }
    return Status(StatusCode::UnsupportedDType,
                  "fused Q/gate RMSNorm/RoPE supports FP32 and BF16");
}

} // namespace hybridai::hip_fused
