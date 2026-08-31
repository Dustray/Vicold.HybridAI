#include "backends/hip/hip_kernels.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace hybridai::hip_kernels {
namespace {

constexpr int kBlockSize = 256;

Status hip_status(hipError_t error, const char* operation) {
    if (error == hipSuccess) return Status::OK();
    return Status(StatusCode::BackendError,
                  std::string(operation) + ": " + hipGetErrorString(error));
}

hipStream_t as_stream(void* stream_handle) {
    return reinterpret_cast<hipStream_t>(stream_handle);
}

template <typename T>
__device__ float load_value(const T* data, int64_t index) {
    return static_cast<float>(data[index]);
}

template <typename T>
__device__ void store_value(T* data, int64_t index, float value) {
    data[index] = static_cast<T>(value);
}

template <typename Dst, typename Src>
__global__ void cast_kernel(Dst* dst, const Src* src, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index < count) store_value(dst, index, load_value(src, index));
}

template <typename T>
__global__ void add_kernel(T* dst, const T* lhs, const T* rhs, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index < count) {
        store_value(dst, index,
                    load_value(lhs, index) + load_value(rhs, index));
    }
}

template <typename T>
__global__ void add_row_bias_kernel(T* dst, const T* bias, int64_t count,
                                    int64_t columns) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index < count) {
        store_value(dst, index,
                    load_value(dst, index) + load_value(bias, index % columns));
    }
}

constexpr int64_t kMaxStridedDimensions = 8;

struct StridedLayout {
    int64_t shape[kMaxStridedDimensions];
    int64_t strides[kMaxStridedDimensions];
    int64_t ndim;
};

template <typename T>
__global__ void strided_copy_kernel(T* dst, const T* src, int64_t count,
                                    StridedLayout layout) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index >= count) return;
    int64_t remaining = index;
    int64_t source_index = 0;
    for (int64_t dimension = layout.ndim - 1; dimension >= 0; --dimension) {
        const int64_t coordinate = remaining % layout.shape[dimension];
        remaining /= layout.shape[dimension];
        source_index += coordinate * layout.strides[dimension];
    }
    dst[index] = src[source_index];
}

template <typename T>
__global__ void embedding_gather_kernel(T* dst, const T* embedding,
                                        const int64_t* ids, int64_t num_ids,
                                        int64_t vocab_size,
                                        int64_t hidden_size) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    const int64_t count = num_ids * hidden_size;
    if (index >= count) return;
    const int64_t row = index / hidden_size;
    const int64_t column = index - row * hidden_size;
    const int64_t token = ids[row];
    dst[index] = token >= 0 && token < vocab_size
                     ? embedding[token * hidden_size + column]
                     : T{};
}

template <typename T>
__global__ void rmsnorm_kernel(T* dst, const T* src, const T* weight,
                               int64_t hidden_size, float eps,
                               bool add_unit_offset) {
    const int64_t row = blockIdx.x;
    const T* row_src = src + row * hidden_size;
    T* row_dst = dst + row * hidden_size;

    float sum = 0.0f;
    for (int64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
        const float value = load_value(row_src, column);
        sum += value * value;
    }

    __shared__ float reduction[kBlockSize];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduction[0] / hidden_size + eps);
    for (int64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
        const float scale = load_value(weight, column) +
                            (add_unit_offset ? 1.0f : 0.0f);
        store_value(row_dst, column,
                    load_value(row_src, column) * inv_rms * scale);
    }
}

__device__ float apply_unary(float value, int op, float param) {
    switch (op) {
        case 0: return fmaxf(0.0f, value);
        case 1: return value / (1.0f + expf(-value));
        case 2:
            return 0.5f * value *
                   (1.0f + tanhf(0.7978845608f *
                                  (value + 0.0447149983f * value * value *
                                               value)));
        case 3: return value / (1.0f + expf(-param * value));
        default: return value;
    }
}

template <typename T>
__global__ void unary_kernel(T* dst, const T* src, int64_t count, int op,
                             float param) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index < count) {
        store_value(dst, index,
                    apply_unary(load_value(src, index), op, param));
    }
}

template <typename T>
__global__ void silu_mul_kernel(T* dst, const T* gate, const T* up,
                                int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index < count) {
        const float gate_value = load_value(gate, index);
        store_value(dst, index,
                    gate_value / (1.0f + expf(-gate_value)) *
                        load_value(up, index));
    }
}

template <typename T>
__global__ void split_q_gate_kernel(T* query, T* gate, const T* source,
                                    int64_t count, int64_t head_dim) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index < count) {
        const int64_t head_index = index / head_dim;
        const int64_t dimension = index % head_dim;
        const int64_t source_index =
            head_index * (2 * head_dim) + dimension;
        query[index] = source[source_index];
        gate[index] = source[source_index + head_dim];
    }
}

template <typename T>
__global__ void partial_rope_kernel(T* dst, const T* src, int64_t count,
                                    int64_t num_heads, int64_t head_dim,
                                    int64_t rope_head_dim,
                                    int64_t position_offset, float base) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index >= count) return;
    const int64_t dimension = index % head_dim;
    if (dimension >= rope_head_dim) {
        dst[index] = src[index];
        return;
    }
    const int64_t half = rope_head_dim / 2;
    const int64_t pair_dimension = dimension % half;
    const int64_t head_index = index / head_dim;
    const int64_t sequence = head_index / num_heads;
    const int64_t head_base = head_index * head_dim;
    const float theta = static_cast<float>(position_offset + sequence) /
        powf(base, 2.0f * static_cast<float>(pair_dimension) /
                       static_cast<float>(rope_head_dim));
    const float cosine = cosf(theta);
    const float sine = sinf(theta);
    const float x = load_value(src, head_base + pair_dimension);
    const float y = load_value(src, head_base + pair_dimension + half);
    const float value = dimension < half
        ? x * cosine - y * sine
        : y * cosine + x * sine;
    store_value(dst, index, value);
}

template <typename T>
__global__ void causal_gqa_kernel(T* dst, const T* query, const T* key,
                                  const T* value, const T* gate,
                                  int64_t seq_len,
                                  int64_t num_query_heads,
                                  int64_t num_kv_heads, int64_t head_dim) {
    const int64_t query_head = blockIdx.x;
    const int64_t sequence = blockIdx.y;
    const int64_t kv_head =
        query_head / (num_query_heads / num_kv_heads);
    const int64_t query_base =
        (sequence * num_query_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float score_max;
    __shared__ float score_sum;
    if (threadIdx.x == 0) {
        float maximum = -INFINITY;
        for (int64_t source_sequence = 0;
             source_sequence <= sequence; ++source_sequence) {
            const int64_t key_base =
                (source_sequence * num_kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                score += load_value(query, query_base + dimension) *
                         load_value(key, key_base + dimension);
            }
            maximum = fmaxf(maximum, score * scale);
        }
        float sum = 0.0f;
        for (int64_t source_sequence = 0;
             source_sequence <= sequence; ++source_sequence) {
            const int64_t key_base =
                (source_sequence * num_kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                score += load_value(query, query_base + dimension) *
                         load_value(key, key_base + dimension);
            }
            sum += expf(score * scale - maximum);
        }
        score_max = maximum;
        score_sum = sum;
    }
    __syncthreads();

    for (int64_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float output = 0.0f;
        for (int64_t source_sequence = 0;
             source_sequence <= sequence; ++source_sequence) {
            const int64_t key_base =
                (source_sequence * num_kv_heads + kv_head) * head_dim;
            const int64_t value_base = key_base;
            float score = 0.0f;
            for (int64_t inner = 0; inner < head_dim; ++inner) {
                score += load_value(query, query_base + inner) *
                         load_value(key, key_base + inner);
            }
            const float probability =
                expf(score * scale - score_max) / score_sum;
            output += probability *
                      load_value(value, value_base + dimension);
        }
        if (gate != nullptr) {
            const float gate_value = load_value(gate, query_base + dimension);
            output *= 1.0f / (1.0f + expf(-gate_value));
        }
        store_value(dst, query_base + dimension, output);
    }
}

template <typename T>
__global__ void append_kv_cache_kernel(T* key_cache, T* value_cache,
                                       const T* key, const T* value,
                                       int64_t count, int64_t row_width,
                                       int64_t cache_offset) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (index >= count) return;
    const int64_t token = index / row_width;
    const int64_t column = index % row_width;
    const int64_t cache_index = (cache_offset + token) * row_width + column;
    key_cache[cache_index] = key[index];
    value_cache[cache_index] = value[index];
}

template <typename T>
__global__ void cached_gqa_kernel(T* dst, const T* query,
                                  const T* key_cache,
                                  const T* value_cache, const T* gate,
                                  int64_t query_len, int64_t cache_len,
                                  int64_t num_query_heads,
                                  int64_t num_kv_heads, int64_t head_dim) {
    const int64_t query_head = blockIdx.x;
    const int64_t query_sequence = blockIdx.y;
    const int64_t kv_head =
        query_head / (num_query_heads / num_kv_heads);
    const int64_t query_base =
        (query_sequence * num_query_heads + query_head) * head_dim;
    const int64_t query_position = cache_len - query_len + query_sequence;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float score_max;
    __shared__ float score_sum;
    if (threadIdx.x == 0) {
        float maximum = -INFINITY;
        for (int64_t source_sequence = 0;
             source_sequence <= query_position; ++source_sequence) {
            const int64_t key_base =
                (source_sequence * num_kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                score += load_value(query, query_base + dimension) *
                         load_value(key_cache, key_base + dimension);
            }
            maximum = fmaxf(maximum, score * scale);
        }
        float sum = 0.0f;
        for (int64_t source_sequence = 0;
             source_sequence <= query_position; ++source_sequence) {
            const int64_t key_base =
                (source_sequence * num_kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
                score += load_value(query, query_base + dimension) *
                         load_value(key_cache, key_base + dimension);
            }
            sum += expf(score * scale - maximum);
        }
        score_max = maximum;
        score_sum = sum;
    }
    __syncthreads();

    for (int64_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
        float output = 0.0f;
        for (int64_t source_sequence = 0;
             source_sequence <= query_position; ++source_sequence) {
            const int64_t key_base =
                (source_sequence * num_kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int64_t inner = 0; inner < head_dim; ++inner) {
                score += load_value(query, query_base + inner) *
                         load_value(key_cache, key_base + inner);
            }
            const float probability =
                expf(score * scale - score_max) / score_sum;
            output += probability *
                      load_value(value_cache, key_base + dimension);
        }
        if (gate != nullptr) {
            const float gate_value = load_value(gate, query_base + dimension);
            output *= 1.0f / (1.0f + expf(-gate_value));
        }
        store_value(dst, query_base + dimension, output);
    }
}

// Decode-optimized GQA for fixed-width heads. One block owns one query
// head, computes each Q.K score cooperatively exactly once, and folds softmax
// and P.V into a single online-softmax pass. The generic kernel above computes
// the same Q.K once per output dimension, which is especially expensive for
// single-token decode.
template <typename T, int64_t HeadDim>
__global__ void cached_gqa_fixed_head_kernel(
    T* dst, const T* query, const T* key_cache, const T* value_cache,
    const T* gate, int64_t query_len, int64_t cache_len,
    int64_t num_query_heads, int64_t num_kv_heads) {
    const int64_t dimension = threadIdx.x;
    const int64_t query_head = blockIdx.x;
    const int64_t query_sequence = blockIdx.y;
    const int64_t kv_head =
        query_head / (num_query_heads / num_kv_heads);
    const int64_t query_base =
        (query_sequence * num_query_heads + query_head) * HeadDim;
    const int64_t query_position =
        cache_len - query_len + query_sequence;
    const float scale = rsqrtf(static_cast<float>(HeadDim));

    const float query_value = load_value(query, query_base + dimension);
    float output = 0.0f;

    // gfx936 uses 64-lane wavefronts, so a 256-thread block produces four
    // partial sums. Size eight also keeps this valid for 32-lane targets.
    __shared__ float wave_sums[HeadDim / 32];
    __shared__ float online_max;
    __shared__ float online_sum;
    __shared__ float previous_scale;
    __shared__ float current_scale;
    if (dimension == 0) {
        online_max = -INFINITY;
        online_sum = 0.0f;
    }
    __syncthreads();

    for (int64_t source_sequence = 0;
         source_sequence <= query_position; ++source_sequence) {
        const int64_t cache_base =
            (source_sequence * num_kv_heads + kv_head) * HeadDim;
        float partial =
            query_value * load_value(key_cache, cache_base + dimension);
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            partial += __shfl_down(partial, offset);
        }
        const int lane = static_cast<int>(dimension) % warpSize;
        const int wave = static_cast<int>(dimension) / warpSize;
        if (lane == 0) wave_sums[wave] = partial;
        __syncthreads();

        if (dimension == 0) {
            float dot = 0.0f;
            for (int wave_index = 0;
                  wave_index < HeadDim / warpSize; ++wave_index) {
                dot += wave_sums[wave_index];
            }
            const float score = dot * scale;
            const float next_max = fmaxf(online_max, score);
            previous_scale = expf(online_max - next_max);
            current_scale = expf(score - next_max);
            online_sum = online_sum * previous_scale + current_scale;
            online_max = next_max;
        }
        __syncthreads();

        output = output * previous_scale +
                 current_scale *
                     load_value(value_cache, cache_base + dimension);
        // Prevent the next score from overwriting shared softmax state before
        // every lane has consumed it.
        __syncthreads();
    }

    output /= online_sum;
    if (gate != nullptr) {
        const float gate_value = load_value(gate, query_base + dimension);
        output *= 1.0f / (1.0f + expf(-gate_value));
    }
    store_value(dst, query_base + dimension, output);
}

template <typename T>
__global__ void causal_conv1d_silu_kernel(
    T* dst, T* conv_state, const T* src, const T* weight,
    int64_t token_count, int64_t channels, int64_t kernel_size) {
    const int64_t channel = blockIdx.x;
    if (channel >= channels || threadIdx.x != 0) return;
    const int64_t history = kernel_size - 1;
    T* channel_state = conv_state + channel * history;
    const T* channel_weight = weight + channel * kernel_size;
    for (int64_t token = 0; token < token_count; ++token) {
        float sum = 0.0f;
        for (int64_t index = 0; index < history; ++index) {
            sum += load_value(channel_state, index) *
                   load_value(channel_weight, index);
        }
        const float current = load_value(src, token * channels + channel);
        sum += current * load_value(channel_weight, history);
        store_value(dst, token * channels + channel,
                    sum / (1.0f + expf(-sum)));
        for (int64_t index = 0; index + 1 < history; ++index)
            channel_state[index] = channel_state[index + 1];
        if (history > 0) store_value(channel_state, history - 1, current);
    }
}

template <typename T>
__global__ void deltanet_grouped_conv_kernel(
    T* query, T* key, T* value, T* conv_state, const T* qkv,
    const T* weight, int64_t token_count, int64_t num_qk_heads,
    int64_t num_value_heads, int64_t key_head_dim,
    int64_t value_head_dim, int64_t kernel_size) {
    const int64_t flat_channel = blockIdx.x;
    const int64_t key_width = num_qk_heads * key_head_dim;
    const int64_t value_width = num_value_heads * value_head_dim;
    const int64_t qkv_width = key_width * 2 + value_width;
    if (flat_channel >= qkv_width || threadIdx.x != 0) return;
    // Qwen3.5 stores both in_proj_qkv output and depthwise Conv1D weights in
    // flat [all Q][all K][all V] channel order. No per-QK-head regrouping is
    // applied here; older Qwen3-Next checkpoints require a separate layout
    // conversion path.
    const int64_t history = kernel_size - 1;
    T* channel_state = conv_state + flat_channel * history;
    const T* channel_weight = weight + flat_channel * kernel_size;
    for (int64_t token = 0; token < token_count; ++token) {
        float sum = 0.0f;
        for (int64_t index = 0; index < history; ++index)
            sum += load_value(channel_state, index) *
                   load_value(channel_weight, index);
        const float current = load_value(
            qkv, token * qkv_width + flat_channel);
        sum += current * load_value(channel_weight, history);
        const float activated = sum / (1.0f + expf(-sum));
        if (flat_channel < key_width)
            store_value(query, token * key_width + flat_channel, activated);
        else if (flat_channel < key_width * 2)
            store_value(key, token * key_width + flat_channel - key_width,
                        activated);
        else
            store_value(value,
                        token * value_width + flat_channel - key_width * 2,
                        activated);
        for (int64_t index = 0; index + 1 < history; ++index)
            channel_state[index] = channel_state[index + 1];
        if (history > 0) store_value(channel_state, history - 1, current);
    }
}

template <typename T>
__global__ void deltanet_recurrent_kernel(
    T* dst, float* recurrent_state, const T* query, const T* key,
    const T* value, const T* a, const T* beta, const T* a_log,
    const T* dt_bias, const T* norm_weight, const T* z,
    int64_t token_count, int64_t num_qk_heads,
    int64_t num_value_heads, int64_t key_head_dim,
    int64_t value_head_dim, int64_t norm_weight_count, float eps) {
    const int64_t value_head = blockIdx.x;
    if (value_head >= num_value_heads) return;
    const int64_t lane = threadIdx.x;
    const int64_t heads_per_group = num_value_heads / num_qk_heads;
    const int64_t qk_head = value_head / heads_per_group;
    float* state = recurrent_state +
        value_head * key_head_dim * value_head_dim;
    const float query_scale = rsqrtf(static_cast<float>(key_head_dim));

    __shared__ float q_scale_shared;
    __shared__ float k_scale_shared;
    __shared__ float decay_shared;
    __shared__ float beta_shared;
    __shared__ float rms_partial[kBlockSize];

    for (int64_t token = 0; token < token_count; ++token) {
        const T* q = query +
            (token * num_qk_heads + qk_head) * key_head_dim;
        const T* k = key +
            (token * num_qk_heads + qk_head) * key_head_dim;
        const T* v = value +
            (token * num_value_heads + value_head) * value_head_dim;
        T* out = dst +
            (token * num_value_heads + value_head) * value_head_dim;
        const int64_t scalar_index = token * num_value_heads + value_head;
        float q_square_sum = 0.0f, k_square_sum = 0.0f;
        for (int64_t i = lane; i < key_head_dim; i += blockDim.x) {
            const float q_value = load_value(q, i);
            const float k_value = load_value(k, i);
            q_square_sum += q_value * q_value;
            k_square_sum += k_value * k_value;
        }
        rms_partial[lane] = q_square_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) rms_partial[lane] += rms_partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0)
            q_scale_shared = rsqrtf(rms_partial[0] + 1.0e-6f) * query_scale;
        rms_partial[lane] = k_square_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) rms_partial[lane] += rms_partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            k_scale_shared = rsqrtf(rms_partial[0] + 1.0e-6f);
            const float decay_arg = load_value(dt_bias, value_head) +
                                    load_value(a, scalar_index);
            const float softplus = fmaxf(decay_arg, 0.0f) +
                                   log1pf(expf(-fabsf(decay_arg)));
            decay_shared = expf(-expf(load_value(a_log, value_head)) *
                                softplus);
            const float beta_logit = load_value(beta, scalar_index);
            beta_shared = 1.0f / (1.0f + expf(-beta_logit));
        }
        __syncthreads();

        float rms_sum = 0.0f;
        for (int64_t j = lane; j < value_head_dim; j += blockDim.x) {
            float retrieved = 0.0f;
            for (int64_t i = 0; i < key_head_dim; ++i) {
                float& state_value = state[i * value_head_dim + j];
                state_value *= decay_shared;
                retrieved += load_value(k, i) * k_scale_shared * state_value;
            }
            const float corrected = load_value(v, j) - retrieved;
            for (int64_t i = 0; i < key_head_dim; ++i)
                state[i * value_head_dim + j] +=
                    beta_shared * load_value(k, i) * k_scale_shared * corrected;
            float output = 0.0f;
            for (int64_t i = 0; i < key_head_dim; ++i)
                output += load_value(q, i) * q_scale_shared *
                          state[i * value_head_dim + j];
            store_value(out, j, output);
            rms_sum += output * output;
        }
        rms_partial[lane] = rms_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (lane < stride) rms_partial[lane] += rms_partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0)
            rms_partial[0] = rsqrtf(rms_partial[0] /
                                    static_cast<float>(value_head_dim) + eps);
        __syncthreads();
        const T* gate = z +
            (token * num_value_heads + value_head) * value_head_dim;
        for (int64_t j = lane; j < value_head_dim; j += blockDim.x) {
            const float gate_value = load_value(gate, j);
            const float silu_gate = gate_value /
                                    (1.0f + expf(-gate_value));
            const int64_t norm_index = norm_weight_count == num_value_heads
                                           ? value_head
                                           : j;
            store_value(out, j, load_value(out, j) * rms_partial[0] *
                         load_value(norm_weight, norm_index) * silu_gate);
        }
        __syncthreads();
    }
}

// Qwen3.5/Qwen3.8 decode specialization. The recurrent state is 128x128,
// and decode supplies one token at a time. The generic kernel launches 256
// threads although only 128 output columns are active. This specialization
// keeps 128 active lanes and uses wavefront shuffles only for reductions.
template <typename T>
__global__ void deltanet_recurrent_decode128_kernel(
    T* dst, float* recurrent_state, const T* query, const T* key,
    const T* value, const T* a, const T* beta, const T* a_log,
    const T* dt_bias, const T* norm_weight, const T* z,
    int64_t num_qk_heads, int64_t num_value_heads, float eps) {
    constexpr int64_t kKeyDim = 128;
    constexpr int64_t kValueDim = 128;
    constexpr int kWaveCount = 128 / warpSize;
    const int64_t value_head = blockIdx.x;
    const int64_t lane = threadIdx.x;
    if (value_head >= num_value_heads) return;

    const int64_t heads_per_group = num_value_heads / num_qk_heads;
    const int64_t qk_head = value_head / heads_per_group;
    const T* q = query + qk_head * kKeyDim;
    const T* k = key + qk_head * kKeyDim;
    const T* v = value + value_head * kValueDim;
    T* out = dst + value_head * kValueDim;
    float* state = recurrent_state + value_head * kKeyDim * kValueDim;

    __shared__ float wave_sums[8];
    __shared__ float q_scale_shared;
    __shared__ float k_scale_shared;
    __shared__ float decay_shared;
    __shared__ float beta_shared;

    float q_partial = 0.0f;
    float k_partial = 0.0f;
    for (int64_t i = lane; i < kKeyDim; i += blockDim.x) {
        const float q_value = load_value(q, i);
        const float k_value = load_value(k, i);
        q_partial += q_value * q_value;
        k_partial += k_value * k_value;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        q_partial += __shfl_down(q_partial, offset);
        k_partial += __shfl_down(k_partial, offset);
    }
    const int wave = static_cast<int>(lane) / warpSize;
    const int lane_in_wave = static_cast<int>(lane) % warpSize;
    if (lane_in_wave == 0) {
        wave_sums[wave] = q_partial;
        wave_sums[wave + kWaveCount] = k_partial;
    }
    __syncthreads();

    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int wave_index = 0; wave_index < kWaveCount; ++wave_index) {
            q_sum += wave_sums[wave_index];
            k_sum += wave_sums[wave_index + kWaveCount];
        }
        q_scale_shared = rsqrtf(q_sum + 1.0e-6f) * rsqrtf(128.0f);
        k_scale_shared = rsqrtf(k_sum + 1.0e-6f);
        const float decay_arg = load_value(dt_bias, value_head) +
                                load_value(a, value_head);
        const float softplus = fmaxf(decay_arg, 0.0f) +
                               log1pf(expf(-fabsf(decay_arg)));
        decay_shared = expf(-expf(load_value(a_log, value_head)) * softplus);
        const float beta_logit = load_value(beta, value_head);
        beta_shared = 1.0f / (1.0f + expf(-beta_logit));
    }
    __syncthreads();

    const int64_t j = lane;
    float retrieved = 0.0f;
    for (int64_t i = 0; i < kKeyDim; ++i) {
        float& state_value = state[i * kValueDim + j];
        state_value *= decay_shared;
        retrieved += load_value(k, i) * k_scale_shared * state_value;
    }
    const float corrected = load_value(v, j) - retrieved;
    for (int64_t i = 0; i < kKeyDim; ++i) {
        state[i * kValueDim + j] +=
            beta_shared * load_value(k, i) * k_scale_shared * corrected;
    }
    const float q_scale = q_scale_shared;
    float output = 0.0f;
    for (int64_t i = 0; i < kKeyDim; ++i)
        output += load_value(q, i) * q_scale * state[i * kValueDim + j];
    store_value(out, j, output);

    float rms_partial = output * output;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        rms_partial += __shfl_down(rms_partial, offset);
    if (lane_in_wave == 0) wave_sums[wave] = rms_partial;
    __syncthreads();
    if (lane == 0) {
        float rms_sum = 0.0f;
        for (int wave_index = 0; wave_index < kWaveCount; ++wave_index)
            rms_sum += wave_sums[wave_index];
        wave_sums[0] = rsqrtf(rms_sum / static_cast<float>(kValueDim) + eps);
    }
    __syncthreads();

    const T* gate = z + value_head * kValueDim;
    const float gate_value = load_value(gate, j);
    const float silu_gate = gate_value / (1.0f + expf(-gate_value));
    store_value(out, j, load_value(out, j) * wave_sums[0] *
                         load_value(norm_weight, j) * silu_gate);
}

template <typename T>
__global__ void argmax_last_row_kernel(int64_t* dst, const T* src,
                                       int64_t rows, int64_t columns) {
    const T* row = src + (rows - 1) * columns;
    float best_value = -INFINITY;
    int64_t best_index = 0;
    for (int64_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        const float value = load_value(row, column);
        if (value > best_value) {
            best_value = value;
            best_index = column;
        }
    }

    __shared__ float values[kBlockSize];
    __shared__ int64_t indices[kBlockSize];
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            values[threadIdx.x + stride] > values[threadIdx.x]) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *dst = indices[0];
}

dim3 grid_for(int64_t count) {
    return dim3(static_cast<unsigned int>((count + kBlockSize - 1) /
                                          kBlockSize));
}

template <typename T>
Status launch_add(void* dst, const void* lhs, const void* rhs, int64_t count,
                  hipStream_t stream) {
    hipLaunchKernelGGL(add_kernel<T>, grid_for(count), dim3(kBlockSize), 0,
                       stream, static_cast<T*>(dst),
                       static_cast<const T*>(lhs), static_cast<const T*>(rhs),
                       count);
    return hip_status(hipGetLastError(), "launch add kernel");
}

template <typename T>
Status launch_add_row_bias(void* dst, const void* bias, int64_t rows,
                           int64_t columns, hipStream_t stream) {
    const int64_t count = rows * columns;
    hipLaunchKernelGGL(add_row_bias_kernel<T>, grid_for(count),
                       dim3(kBlockSize), 0, stream, static_cast<T*>(dst),
                       static_cast<const T*>(bias), count, columns);
    return hip_status(hipGetLastError(), "launch row bias add kernel");
}

template <typename T>
Status launch_strided_copy(void* dst, const void* src, int64_t count,
                           const StridedLayout& layout, hipStream_t stream) {
    hipLaunchKernelGGL(strided_copy_kernel<T>, grid_for(count),
                       dim3(kBlockSize), 0, stream, static_cast<T*>(dst),
                       static_cast<const T*>(src), count, layout);
    return hip_status(hipGetLastError(), "launch strided copy kernel");
}

template <typename T>
Status launch_embedding(void* dst, const void* embedding, const int64_t* ids,
                        int64_t num_ids, int64_t vocab_size,
                        int64_t hidden_size, hipStream_t stream) {
    hipLaunchKernelGGL(embedding_gather_kernel<T>,
                       grid_for(num_ids * hidden_size), dim3(kBlockSize), 0,
                       stream, static_cast<T*>(dst),
                       static_cast<const T*>(embedding), ids, num_ids,
                       vocab_size, hidden_size);
    return hip_status(hipGetLastError(), "launch embedding gather kernel");
}

template <typename T>
Status launch_rmsnorm(void* dst, const void* src, const void* weight,
                      int64_t rows, int64_t hidden_size, float eps,
                      bool add_unit_offset, hipStream_t stream) {
    hipLaunchKernelGGL(rmsnorm_kernel<T>, dim3(static_cast<unsigned int>(rows)),
                       dim3(kBlockSize), 0, stream, static_cast<T*>(dst),
                       static_cast<const T*>(src),
                       static_cast<const T*>(weight), hidden_size, eps,
                       add_unit_offset);
    return hip_status(hipGetLastError(), "launch RMSNorm kernel");
}

template <typename T>
Status launch_unary(void* dst, const void* src, int64_t count, int op,
                    float param, hipStream_t stream) {
    hipLaunchKernelGGL(unary_kernel<T>, grid_for(count), dim3(kBlockSize), 0,
                       stream, static_cast<T*>(dst),
                       static_cast<const T*>(src), count, op, param);
    return hip_status(hipGetLastError(), "launch unary kernel");
}

template <typename T>
Status launch_silu_mul(void* dst, const void* gate, const void* up,
                       int64_t count, hipStream_t stream) {
    hipLaunchKernelGGL(silu_mul_kernel<T>, grid_for(count), dim3(kBlockSize), 0,
                       stream, static_cast<T*>(dst),
                       static_cast<const T*>(gate), static_cast<const T*>(up),
                       count);
    return hip_status(hipGetLastError(), "launch fused SiLU multiply kernel");
}

template <typename T>
Status launch_split_q_gate(void* query, void* gate, const void* source,
                           int64_t rows, int64_t heads, int64_t head_dim,
                           hipStream_t stream) {
    const int64_t count = rows * heads * head_dim;
    hipLaunchKernelGGL(split_q_gate_kernel<T>, grid_for(count),
                       dim3(kBlockSize), 0, stream, static_cast<T*>(query),
                       static_cast<T*>(gate), static_cast<const T*>(source),
                       count, head_dim);
    return hip_status(hipGetLastError(), "launch query/gate split kernel");
}

template <typename T>
Status launch_partial_rope(void* dst, const void* src, int64_t seq_len,
                           int64_t num_heads, int64_t head_dim,
                           int64_t rope_head_dim, int64_t position_offset,
                           float base, hipStream_t stream) {
    const int64_t count = seq_len * num_heads * head_dim;
    hipLaunchKernelGGL(partial_rope_kernel<T>, grid_for(count),
                       dim3(kBlockSize), 0, stream, static_cast<T*>(dst),
                       static_cast<const T*>(src), count, num_heads, head_dim,
                       rope_head_dim, position_offset, base);
    return hip_status(hipGetLastError(), "launch partial RoPE kernel");
}

template <typename T>
Status launch_causal_gqa(void* dst, const void* query, const void* key,
                         const void* value, const void* gate,
                         int64_t seq_len, int64_t num_query_heads,
                         int64_t num_kv_heads, int64_t head_dim,
                         hipStream_t stream) {
    hipLaunchKernelGGL(causal_gqa_kernel<T>,
                       dim3(static_cast<unsigned int>(num_query_heads),
                            static_cast<unsigned int>(seq_len)),
                       dim3(kBlockSize), 0, stream, static_cast<T*>(dst),
                       static_cast<const T*>(query),
                       static_cast<const T*>(key),
                       static_cast<const T*>(value),
                       static_cast<const T*>(gate), seq_len, num_query_heads,
                       num_kv_heads, head_dim);
    return hip_status(hipGetLastError(), "launch causal GQA kernel");
}

template <typename T>
Status launch_append_kv_cache(void* key_cache, void* value_cache,
                              const void* key, const void* value,
                              int64_t token_count, int64_t num_kv_heads,
                              int64_t head_dim, int64_t cache_offset,
                              hipStream_t stream) {
    const int64_t row_width = num_kv_heads * head_dim;
    const int64_t count = token_count * row_width;
    hipLaunchKernelGGL(append_kv_cache_kernel<T>, grid_for(count),
                       dim3(kBlockSize), 0, stream,
                       static_cast<T*>(key_cache),
                       static_cast<T*>(value_cache),
                       static_cast<const T*>(key),
                       static_cast<const T*>(value), count, row_width,
                       cache_offset);
    return hip_status(hipGetLastError(), "launch KV cache append kernel");
}

template <typename T>
Status launch_cached_gqa(void* dst, const void* query, const void* key_cache,
                         const void* value_cache, const void* gate,
                         int64_t query_len, int64_t cache_len,
                         int64_t num_query_heads, int64_t num_kv_heads,
                         int64_t head_dim, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(num_query_heads),
                    static_cast<unsigned int>(query_len));
    if (head_dim == 128) {
        hipLaunchKernelGGL((cached_gqa_fixed_head_kernel<T, 128>), grid,
                           dim3(128), 0, stream,
                           static_cast<T*>(dst),
                           static_cast<const T*>(query),
                           static_cast<const T*>(key_cache),
                           static_cast<const T*>(value_cache),
                           static_cast<const T*>(gate), query_len, cache_len,
                           num_query_heads, num_kv_heads);
    } else if (head_dim == 256) {
        hipLaunchKernelGGL((cached_gqa_fixed_head_kernel<T, 256>), grid,
                           dim3(256), 0, stream,
                           static_cast<T*>(dst),
                           static_cast<const T*>(query),
                           static_cast<const T*>(key_cache),
                           static_cast<const T*>(value_cache),
                           static_cast<const T*>(gate), query_len, cache_len,
                           num_query_heads, num_kv_heads);
    } else {
        hipLaunchKernelGGL(cached_gqa_kernel<T>, grid, dim3(kBlockSize), 0,
                           stream, static_cast<T*>(dst),
                           static_cast<const T*>(query),
                           static_cast<const T*>(key_cache),
                           static_cast<const T*>(value_cache),
                           static_cast<const T*>(gate), query_len, cache_len,
                           num_query_heads, num_kv_heads, head_dim);
    }
    return hip_status(hipGetLastError(), "launch cached GQA kernel");
}

template <typename T>
Status launch_causal_conv1d_silu(
    void* dst, void* conv_state, const void* src, const void* weight,
    int64_t token_count, int64_t channels, int64_t kernel_size,
    hipStream_t stream) {
    hipLaunchKernelGGL(causal_conv1d_silu_kernel<T>,
                       dim3(static_cast<unsigned int>(channels)), dim3(1), 0,
                       stream, static_cast<T*>(dst),
                       static_cast<T*>(conv_state), static_cast<const T*>(src),
                       static_cast<const T*>(weight), token_count, channels,
                       kernel_size);
    return hip_status(hipGetLastError(), "launch causal conv1d SiLU kernel");
}

template <typename T>
Status launch_deltanet_grouped_conv(
    void* query, void* key, void* value, void* conv_state,
    const void* grouped_qkv, const void* weight, int64_t token_count,
    int64_t num_qk_heads, int64_t num_value_heads, int64_t key_head_dim,
    int64_t value_head_dim, int64_t kernel_size, hipStream_t stream) {
    const int64_t channels = num_qk_heads * key_head_dim * 2 +
                             num_value_heads * value_head_dim;
    hipLaunchKernelGGL(deltanet_grouped_conv_kernel<T>,
                       dim3(static_cast<unsigned int>(channels)), dim3(1), 0,
                       stream, static_cast<T*>(query), static_cast<T*>(key),
                       static_cast<T*>(value), static_cast<T*>(conv_state),
                       static_cast<const T*>(grouped_qkv),
                       static_cast<const T*>(weight), token_count,
                       num_qk_heads, num_value_heads, key_head_dim,
                       value_head_dim, kernel_size);
    return hip_status(hipGetLastError(),
                      "launch grouped DeltaNet conv kernel");
}

template <typename T>
Status launch_deltanet_recurrent(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, int64_t norm_weight_count,
    float eps,
    hipStream_t stream) {
    if (token_count == 1 && key_head_dim == 128 && value_head_dim == 128) {
        hipLaunchKernelGGL(
            deltanet_recurrent_decode128_kernel<T>,
            dim3(static_cast<unsigned int>(num_value_heads)), dim3(128), 0,
            stream, static_cast<T*>(dst), static_cast<float*>(recurrent_state),
            static_cast<const T*>(query), static_cast<const T*>(key),
            static_cast<const T*>(value), static_cast<const T*>(a),
            static_cast<const T*>(beta), static_cast<const T*>(a_log),
            static_cast<const T*>(dt_bias), static_cast<const T*>(norm_weight),
            static_cast<const T*>(z), num_qk_heads, num_value_heads, eps);
        return hip_status(hipGetLastError(),
                          "launch DeltaNet decode128 kernel");
    }
    hipLaunchKernelGGL(deltanet_recurrent_kernel<T>,
                       dim3(static_cast<unsigned int>(num_value_heads)),
                       dim3(kBlockSize), 0, stream, static_cast<T*>(dst),
                       static_cast<float*>(recurrent_state),
                       static_cast<const T*>(query), static_cast<const T*>(key),
                       static_cast<const T*>(value), static_cast<const T*>(a),
                       static_cast<const T*>(beta),
                       static_cast<const T*>(a_log),
                       static_cast<const T*>(dt_bias),
                       static_cast<const T*>(norm_weight),
                       static_cast<const T*>(z), token_count, num_qk_heads,
                       num_value_heads, key_head_dim, value_head_dim,
                       norm_weight_count, eps);
    return hip_status(hipGetLastError(), "launch DeltaNet recurrent kernel");
}

template <typename T>
Status launch_deltanet_recurrent_decode128_for_test(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z,
    int64_t num_qk_heads, int64_t num_value_heads, float eps,
    hipStream_t stream) {
    hipLaunchKernelGGL(
        deltanet_recurrent_decode128_kernel<T>,
        dim3(static_cast<unsigned int>(num_value_heads)), dim3(128), 0,
        stream, static_cast<T*>(dst), static_cast<float*>(recurrent_state),
        static_cast<const T*>(query), static_cast<const T*>(key),
        static_cast<const T*>(value), static_cast<const T*>(a),
        static_cast<const T*>(beta), static_cast<const T*>(a_log),
        static_cast<const T*>(dt_bias), static_cast<const T*>(norm_weight),
        static_cast<const T*>(z), num_qk_heads, num_value_heads, eps);
    return hip_status(hipGetLastError(),
                      "launch DeltaNet decode128 test kernel");
}

template <typename T>
Status launch_argmax(int64_t* dst, const void* src, int64_t rows,
                     int64_t columns, hipStream_t stream) {
    hipLaunchKernelGGL(argmax_last_row_kernel<T>, dim3(1), dim3(kBlockSize), 0,
                       stream, dst, static_cast<const T*>(src), rows, columns);
    return hip_status(hipGetLastError(), "launch argmax kernel");
}

} // namespace

Status cast(void* dst, const void* src, DType dst_type, DType src_type,
            int64_t count, void* stream_handle) {
    if (dst == nullptr || src == nullptr || count <= 0) {
        return Status(StatusCode::InvalidArgument, "Invalid HIP cast arguments");
    }
    const dim3 grid = grid_for(count);
    const dim3 block(kBlockSize);
    const hipStream_t stream = as_stream(stream_handle);
#define LAUNCH_CAST(DST, SRC)                                                   \
    hipLaunchKernelGGL((cast_kernel<DST, SRC>), grid, block, 0, stream,         \
                       static_cast<DST*>(dst), static_cast<const SRC*>(src),     \
                       count)
    if (dst_type == DType::FP32 && src_type == DType::BF16) {
        LAUNCH_CAST(float, hip_bfloat16);
    } else if (dst_type == DType::BF16 && src_type == DType::FP32) {
        LAUNCH_CAST(hip_bfloat16, float);
    } else if (dst_type == DType::FP32 && src_type == DType::FP16) {
        LAUNCH_CAST(float, __half);
    } else if (dst_type == DType::FP16 && src_type == DType::FP32) {
        LAUNCH_CAST(__half, float);
    } else {
        return Status(StatusCode::UnsupportedDType,
                      "Unsupported HIP cast dtype pair");
    }
#undef LAUNCH_CAST
    return hip_status(hipGetLastError(), "launch cast kernel");
}

Status add(void* dst, const void* lhs, const void* rhs, DType dtype,
           int64_t count, void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_add<float>(dst, lhs, rhs, count, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_add<hip_bfloat16>(dst, lhs, rhs, count,
                                        as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType, "Unsupported HIP add dtype");
}

Status add_row_bias(void* dst, const void* bias, DType dtype, int64_t rows,
                    int64_t columns, void* stream_handle) {
    if (dst == nullptr || bias == nullptr || rows <= 0 || columns <= 0) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid HIP row bias add arguments");
    }
    if (dtype == DType::FP32)
        return launch_add_row_bias<float>(dst, bias, rows, columns,
                                          as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_add_row_bias<hip_bfloat16>(
            dst, bias, rows, columns, as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP row bias add dtype");
}

Status strided_copy(void* dst, const void* src, DType dtype, int64_t count,
                    const int64_t* shape, const int64_t* strides,
                    int64_t ndim, void* stream_handle) {
    if (dst == nullptr || src == nullptr || count <= 0 || shape == nullptr ||
        strides == nullptr || ndim <= 0 || ndim > kMaxStridedDimensions) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid HIP strided copy arguments");
    }
    StridedLayout layout{};
    layout.ndim = ndim;
    for (int64_t dimension = 0; dimension < ndim; ++dimension) {
        if (shape[dimension] <= 0 || strides[dimension] < 0) {
            return Status(StatusCode::InvalidArgument,
                          "Invalid HIP strided layout");
        }
        layout.shape[dimension] = shape[dimension];
        layout.strides[dimension] = strides[dimension];
    }
    if (dtype == DType::FP32)
        return launch_strided_copy<float>(dst, src, count, layout,
                                          as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_strided_copy<hip_bfloat16>(
            dst, src, count, layout, as_stream(stream_handle));
    if (dtype == DType::FP16)
        return launch_strided_copy<__half>(dst, src, count, layout,
                                           as_stream(stream_handle));
    if (dtype == DType::INT8 || dtype == DType::FP8_E4M3)
        return launch_strided_copy<uint8_t>(dst, src, count, layout,
                                            as_stream(stream_handle));
    if (dtype == DType::INT32)
        return launch_strided_copy<int32_t>(dst, src, count, layout,
                                            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP strided copy dtype");
}

Status embedding_gather(void* dst, const void* embedding, const int64_t* ids,
                        DType dtype, int64_t num_ids, int64_t vocab_size,
                        int64_t hidden_size, void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_embedding<float>(dst, embedding, ids, num_ids, vocab_size,
                                       hidden_size, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_embedding<hip_bfloat16>(
            dst, embedding, ids, num_ids, vocab_size, hidden_size,
            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP embedding dtype");
}

Status rmsnorm(void* dst, const void* src, const void* weight, DType dtype,
               int64_t rows, int64_t hidden_size, float eps,
               bool add_unit_offset, void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_rmsnorm<float>(dst, src, weight, rows, hidden_size, eps,
                                     add_unit_offset, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_rmsnorm<hip_bfloat16>(
            dst, src, weight, rows, hidden_size, eps, add_unit_offset,
            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP RMSNorm dtype");
}

Status unary(void* dst, const void* src, DType dtype, int64_t count, int op,
             float param, void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_unary<float>(dst, src, count, op, param,
                                   as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_unary<hip_bfloat16>(dst, src, count, op, param,
                                          as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP unary dtype");
}

Status silu_mul(void* dst, const void* gate, const void* up, DType dtype,
                int64_t count, void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_silu_mul<float>(dst, gate, up, count,
                                      as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_silu_mul<hip_bfloat16>(dst, gate, up, count,
                                             as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP fused SiLU multiply dtype");
}

Status split_q_gate(void* query, void* gate, const void* source, DType dtype,
                    int64_t rows, int64_t heads, int64_t head_dim,
                    void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_split_q_gate<float>(query, gate, source, rows, heads,
                                          head_dim, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_split_q_gate<hip_bfloat16>(
            query, gate, source, rows, heads, head_dim,
            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP query/gate split dtype");
}

Status partial_rope(void* dst, const void* src, DType dtype, int64_t seq_len,
                    int64_t num_heads, int64_t head_dim,
                    int64_t rope_head_dim, int64_t position_offset,
                    float base, void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_partial_rope<float>(
            dst, src, seq_len, num_heads, head_dim, rope_head_dim,
            position_offset, base, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_partial_rope<hip_bfloat16>(
            dst, src, seq_len, num_heads, head_dim, rope_head_dim,
            position_offset, base, as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP partial RoPE dtype");
}

Status causal_gqa(void* dst, const void* query, const void* key,
                  const void* value, const void* gate, DType dtype,
                  int64_t seq_len, int64_t num_query_heads,
                  int64_t num_kv_heads, int64_t head_dim,
                  void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_causal_gqa<float>(
            dst, query, key, value, gate, seq_len, num_query_heads,
            num_kv_heads, head_dim, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_causal_gqa<hip_bfloat16>(
            dst, query, key, value, gate, seq_len, num_query_heads,
            num_kv_heads, head_dim, as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP causal GQA dtype");
}

Status append_kv_cache(void* key_cache, void* value_cache, const void* key,
                       const void* value, DType dtype, int64_t token_count,
                       int64_t num_kv_heads, int64_t head_dim,
                       int64_t cache_offset, int64_t cache_capacity,
                       void* stream_handle) {
    if (key_cache == nullptr || value_cache == nullptr || key == nullptr ||
        value == nullptr || token_count <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || cache_offset < 0 ||
        cache_offset + token_count > cache_capacity) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid HIP KV cache append arguments");
    }
    if (dtype == DType::FP32)
        return launch_append_kv_cache<float>(
            key_cache, value_cache, key, value, token_count, num_kv_heads,
            head_dim, cache_offset, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_append_kv_cache<hip_bfloat16>(
            key_cache, value_cache, key, value, token_count, num_kv_heads,
            head_dim, cache_offset, as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP KV cache dtype");
}

Status cached_gqa(void* dst, const void* query, const void* key_cache,
                  const void* value_cache, const void* gate, DType dtype,
                  int64_t query_len, int64_t cache_len,
                  int64_t num_query_heads, int64_t num_kv_heads,
                  int64_t head_dim, void* stream_handle) {
    if (dst == nullptr || query == nullptr || key_cache == nullptr ||
        value_cache == nullptr || query_len <= 0 || cache_len < query_len ||
        num_query_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        num_query_heads % num_kv_heads != 0) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid HIP cached GQA arguments");
    }
    if (dtype == DType::FP32)
        return launch_cached_gqa<float>(
            dst, query, key_cache, value_cache, gate, query_len, cache_len,
            num_query_heads, num_kv_heads, head_dim, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_cached_gqa<hip_bfloat16>(
            dst, query, key_cache, value_cache, gate, query_len, cache_len,
            num_query_heads, num_kv_heads, head_dim, as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP cached GQA dtype");
}

Status causal_conv1d_silu(void* dst, void* conv_state, const void* src,
                          const void* weight, DType dtype,
                          int64_t token_count, int64_t channels,
                          int64_t kernel_size, void* stream_handle) {
    if (dst == nullptr || conv_state == nullptr || src == nullptr ||
        weight == nullptr || token_count <= 0 || channels <= 0 ||
        kernel_size <= 1) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid HIP causal conv1d arguments");
    }
    if (dtype == DType::FP32)
        return launch_causal_conv1d_silu<float>(
            dst, conv_state, src, weight, token_count, channels, kernel_size,
            as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_causal_conv1d_silu<hip_bfloat16>(
            dst, conv_state, src, weight, token_count, channels, kernel_size,
            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP causal conv1d dtype");
}

Status deltanet_grouped_conv(
    void* query, void* key, void* value, void* conv_state,
    const void* grouped_qkv, const void* weight, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, int64_t kernel_size,
    void* stream_handle) {
    if (query == nullptr || key == nullptr || value == nullptr ||
        conv_state == nullptr || grouped_qkv == nullptr || weight == nullptr ||
        token_count <= 0 || num_qk_heads <= 0 || num_value_heads <= 0 ||
        num_value_heads % num_qk_heads != 0 || key_head_dim <= 0 ||
        value_head_dim <= 0 || kernel_size <= 1)
        return Status(StatusCode::InvalidArgument,
                      "Invalid grouped DeltaNet conv arguments");
    if (dtype == DType::FP32)
        return launch_deltanet_grouped_conv<float>(
            query, key, value, conv_state, grouped_qkv, weight, token_count,
            num_qk_heads, num_value_heads, key_head_dim, value_head_dim,
            kernel_size, as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_deltanet_grouped_conv<hip_bfloat16>(
            query, key, value, conv_state, grouped_qkv, weight, token_count,
            num_qk_heads, num_value_heads, key_head_dim, value_head_dim,
            kernel_size, as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported grouped DeltaNet conv dtype");
}

Status deltanet_recurrent(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, float eps,
    void* stream_handle) {
    if (dst == nullptr || recurrent_state == nullptr || query == nullptr ||
        key == nullptr || value == nullptr || a == nullptr || beta == nullptr ||
        a_log == nullptr || dt_bias == nullptr || norm_weight == nullptr ||
        z == nullptr || token_count <= 0 || num_qk_heads <= 0 ||
        num_value_heads <= 0 || num_value_heads % num_qk_heads != 0 ||
        key_head_dim <= 0 || value_head_dim <= 0 || eps <= 0.0f) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid HIP DeltaNet recurrent arguments");
    }
    const int64_t norm_weight_count = value_head_dim;
    if (dtype == DType::FP32)
        return launch_deltanet_recurrent<float>(
            dst, recurrent_state, query, key, value, a, beta, a_log, dt_bias,
            norm_weight, z, token_count, num_qk_heads, num_value_heads,
            key_head_dim, value_head_dim, norm_weight_count, eps,
            as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_deltanet_recurrent<hip_bfloat16>(
            dst, recurrent_state, query, key, value, a, beta, a_log, dt_bias,
            norm_weight, z, token_count, num_qk_heads, num_value_heads,
            key_head_dim, value_head_dim, norm_weight_count, eps,
            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP DeltaNet recurrent dtype");
}

Status deltanet_recurrent_generic_for_test(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, float eps,
    void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_deltanet_recurrent<float>(
            dst, recurrent_state, query, key, value, a, beta, a_log, dt_bias,
            norm_weight, z, token_count, num_qk_heads, num_value_heads,
            key_head_dim, value_head_dim, value_head_dim, eps,
            as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_deltanet_recurrent<hip_bfloat16>(
            dst, recurrent_state, query, key, value, a, beta, a_log, dt_bias,
            norm_weight, z, token_count, num_qk_heads, num_value_heads,
            key_head_dim, value_head_dim, value_head_dim, eps,
            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP DeltaNet generic test dtype");
}

Status deltanet_recurrent_decode128_for_test(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z, DType dtype,
    int64_t num_qk_heads, int64_t num_value_heads, float eps,
    void* stream_handle) {
    if (num_qk_heads <= 0 || num_value_heads <= 0 ||
        num_value_heads % num_qk_heads != 0 || eps <= 0.0f)
        return Status(StatusCode::InvalidArgument,
                      "Invalid HIP DeltaNet decode128 test arguments");
    if (dtype == DType::FP32)
        return launch_deltanet_recurrent_decode128_for_test<float>(
            dst, recurrent_state, query, key, value, a, beta, a_log, dt_bias,
            norm_weight, z, num_qk_heads, num_value_heads, eps,
            as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_deltanet_recurrent_decode128_for_test<hip_bfloat16>(
            dst, recurrent_state, query, key, value, a, beta, a_log, dt_bias,
            norm_weight, z, num_qk_heads, num_value_heads, eps,
            as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP DeltaNet decode128 test dtype");
}

Status argmax_last_row(int64_t* dst, const void* src, DType dtype,
                       int64_t rows, int64_t columns, void* stream_handle) {
    if (dtype == DType::FP32)
        return launch_argmax<float>(dst, src, rows, columns,
                                    as_stream(stream_handle));
    if (dtype == DType::BF16)
        return launch_argmax<hip_bfloat16>(dst, src, rows, columns,
                                           as_stream(stream_handle));
    return Status(StatusCode::UnsupportedDType,
                  "Unsupported HIP argmax dtype");
}

} // namespace hybridai::hip_kernels