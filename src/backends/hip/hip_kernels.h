#pragma once

#include "core/dtype.h"
#include "core/status.h"

#include <cstdint>

namespace hybridai::hip_kernels {

// stream_handle is a native hipStream_t stored as void* so HIP headers remain
// confined to the backend implementation and this launcher boundary.
Status cast(void* dst, const void* src, DType dst_type, DType src_type,
            int64_t count, void* stream_handle);
Status add(void* dst, const void* lhs, const void* rhs, DType dtype,
           int64_t count, void* stream_handle);
Status add_row_bias(void* dst, const void* bias, DType dtype, int64_t rows,
                    int64_t columns, void* stream_handle);
Status strided_copy(void* dst, const void* src, DType dtype, int64_t count,
                    const int64_t* shape, const int64_t* strides,
                    int64_t ndim, void* stream_handle);
Status embedding_gather(void* dst, const void* embedding, const int64_t* ids,
                        DType dtype, int64_t num_ids, int64_t vocab_size,
                        int64_t hidden_size, void* stream_handle);
Status rmsnorm(void* dst, const void* src, const void* weight, DType dtype,
               int64_t rows, int64_t hidden_size, float eps,
               bool add_unit_offset, void* stream_handle);
Status unary(void* dst, const void* src, DType dtype, int64_t count, int op,
             float param, void* stream_handle);
Status silu_mul(void* dst, const void* gate, const void* up, DType dtype,
                int64_t count, void* stream_handle);
Status split_q_gate(void* query, void* gate, const void* source, DType dtype,
                    int64_t rows, int64_t heads, int64_t head_dim,
                    void* stream_handle);
Status partial_rope(void* dst, const void* src, DType dtype, int64_t seq_len,
                    int64_t num_heads, int64_t head_dim,
                    int64_t rope_head_dim, int64_t position_offset,
                    float base, void* stream_handle);
Status causal_gqa(void* dst, const void* query, const void* key,
                  const void* value, const void* gate, DType dtype,
                  int64_t seq_len, int64_t num_query_heads,
                  int64_t num_kv_heads, int64_t head_dim,
                  void* stream_handle);
Status append_kv_cache(void* key_cache, void* value_cache, const void* key,
                       const void* value, DType dtype, int64_t token_count,
                       int64_t num_kv_heads, int64_t head_dim,
                       int64_t cache_offset, int64_t cache_capacity,
                       void* stream_handle);
Status cached_gqa(void* dst, const void* query, const void* key_cache,
                  const void* value_cache, const void* gate, DType dtype,
                  int64_t query_len, int64_t cache_len,
                  int64_t num_query_heads, int64_t num_kv_heads,
                  int64_t head_dim, void* stream_handle);
Status causal_conv1d_silu(void* dst, void* conv_state, const void* src,
                          const void* weight, DType dtype,
                          int64_t token_count, int64_t channels,
                          int64_t kernel_size, void* stream_handle);
Status deltanet_grouped_conv(
    void* query, void* key, void* value, void* conv_state,
    const void* grouped_qkv, const void* weight, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, int64_t kernel_size,
    void* stream_handle);
Status deltanet_recurrent(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, float eps,
    void* stream_handle);

// Explicit kernel entry points used by HIP numerical equivalence tests. The
// normal deltanet_recurrent() API remains on the portable generic path.
Status deltanet_recurrent_generic_for_test(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z, DType dtype,
    int64_t token_count, int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, float eps,
    void* stream_handle);
Status deltanet_recurrent_decode128_for_test(
    void* dst, void* recurrent_state, const void* query, const void* key,
    const void* value, const void* a, const void* beta, const void* a_log,
    const void* dt_bias, const void* norm_weight, const void* z, DType dtype,
    int64_t num_qk_heads, int64_t num_value_heads, float eps,
    void* stream_handle);
Status argmax_last_row(int64_t* dst, const void* src, DType dtype,
                       int64_t rows, int64_t columns, void* stream_handle);

} // namespace hybridai::hip_kernels