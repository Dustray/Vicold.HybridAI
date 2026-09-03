#pragma once

#include "backends/interface/backend.h"

#include <memory>

namespace hybridai {

// HIP/ROCm backend. All native HIP API calls are confined to this file and its
// implementation. This header exposes only the abstract Backend interface.
class HipBackend final : public Backend {
public:
    explicit HipBackend(const Device& device);
    ~HipBackend() override;

    const char* name() const noexcept override;
    Device device() const noexcept override;
    bool is_available() const noexcept override;

    std::vector<Device> enumerate_devices() const override;

    std::unique_ptr<Allocator> create_allocator(MemoryType type) override;
    std::shared_ptr<Buffer> create_buffer(size_t size,
                                          MemoryType type) override;
    std::unique_ptr<Stream> create_stream() override;
    std::unique_ptr<Event> create_event() override;

    Status memcpy_h2d(Buffer* dst, const void* src, size_t size,
                      Stream* stream) override;
    Status memcpy_d2h(void* dst, const Buffer* src, size_t size,
                      Stream* stream) override;
    Status memcpy_d2d(Buffer* dst, const Buffer* src, size_t size,
                      Stream* stream) override;
    Status copy(Buffer* dst, const Buffer* src, size_t size,
                Stream* stream) override;
    Status copy_to_offset(Buffer* dst, size_t dst_offset, const Buffer* src,
                          size_t src_offset, size_t size,
                          Stream* stream = nullptr) override;
    Status memset(Buffer* dst, int value, size_t size,
                  Stream* stream) override;

    Status synchronize() override;

    Status gemm(Buffer* c, const Buffer* a, const Buffer* b, DType c_type,
                DType a_type, DType b_type, DType compute_type,
                bool trans_a, bool trans_b, int64_t m, int64_t n, int64_t k,
                float alpha, float beta, Stream* stream) override;

    Status cast(Buffer* dst, const Buffer* src, DType dst_type,
                DType src_type, int64_t count, Stream* stream) override;
    Status add(Buffer* dst, const Buffer* lhs, const Buffer* rhs, DType dtype,
               int64_t count, Stream* stream) override;
    Status add_row_bias(Buffer* dst, const Buffer* bias, DType dtype,
                        int64_t rows, int64_t columns,
                        Stream* stream) override;
    Status strided_copy(Buffer* dst, const Buffer* src, DType dtype,
                        int64_t count, const int64_t* shape,
                        const int64_t* strides, int64_t ndim,
                        Stream* stream) override;
    Status embedding_gather(Buffer* dst, const Buffer* embedding,
                            const Buffer* ids, DType dtype, int64_t num_ids,
                            int64_t vocab_size, int64_t hidden_size,
                            Stream* stream) override;
    Status rmsnorm(Buffer* dst, const Buffer* src, const Buffer* weight,
                   DType dtype, int64_t rows, int64_t hidden_size, float eps,
                   bool add_unit_offset, Stream* stream) override;
    Status unary(Buffer* dst, const Buffer* src, DType dtype, int64_t count,
                 int op, float param, Stream* stream) override;
    Status silu_mul(Buffer* dst, const Buffer* gate, const Buffer* up,
                    DType dtype, int64_t count, Stream* stream) override;
    Status split_q_gate(Buffer* query, Buffer* gate, const Buffer* source,
                        DType dtype, int64_t rows, int64_t heads,
                        int64_t head_dim, Stream* stream) override;
    Status partial_rope(Buffer* dst, const Buffer* src, DType dtype,
                        int64_t seq_len, int64_t num_heads,
                        int64_t head_dim, int64_t rope_head_dim,
                        int64_t position_offset, float base,
                        Stream* stream) override;
    Status qk_rmsnorm_rope(
        Buffer* query_dst, const Buffer* query_src,
        const Buffer* query_weight, Buffer* key_dst, const Buffer* key_src,
        const Buffer* key_weight, DType dtype, int64_t seq_len,
        int64_t num_query_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t rope_head_dim, int64_t position_offset, float rope_base,
        float eps, bool add_unit_offset, Stream* stream = nullptr) override;
    Status q_gate_rmsnorm_rope(
        Buffer* query_dst, Buffer* gate_dst, const Buffer* source,
        const Buffer* query_weight, DType dtype, int64_t seq_len,
        int64_t num_query_heads, int64_t head_dim, int64_t rope_head_dim,
        int64_t position_offset, float rope_base, float eps,
        bool add_unit_offset, Stream* stream = nullptr) override;
    Status causal_gqa(Buffer* dst, const Buffer* query, const Buffer* key,
                      const Buffer* value, const Buffer* gate, DType dtype,
                      int64_t seq_len, int64_t num_query_heads,
                      int64_t num_kv_heads, int64_t head_dim,
                      Stream* stream) override;
    Status append_kv_cache(Buffer* key_cache, Buffer* value_cache,
                           const Buffer* key, const Buffer* value, DType dtype,
                           int64_t token_count, int64_t num_kv_heads,
                           int64_t head_dim, int64_t cache_offset,
                           int64_t cache_capacity,
                           Stream* stream = nullptr) override;
    Status cached_gqa(Buffer* dst, const Buffer* query,
                      const Buffer* key_cache, const Buffer* value_cache,
                      const Buffer* gate, DType dtype, int64_t query_len,
                      int64_t cache_len, int64_t num_query_heads,
                      int64_t num_kv_heads, int64_t head_dim,
                      Stream* stream = nullptr) override;
    Status cached_gqa_with_current_token(
        Buffer* dst, const Buffer* query, const Buffer* key_cache,
        const Buffer* value_cache, const Buffer* current_key,
        const Buffer* current_value, const Buffer* gate, DType dtype,
        int64_t cache_len, int64_t num_query_heads, int64_t num_kv_heads,
        int64_t head_dim, Stream* stream = nullptr) override;
    Status causal_conv1d_silu(Buffer* dst, Buffer* conv_state,
                              const Buffer* src, const Buffer* weight,
                              DType dtype, int64_t token_count,
                              int64_t channels, int64_t kernel_size,
                              Stream* stream = nullptr) override;
    Status deltanet_grouped_conv(
        Buffer* query, Buffer* key, Buffer* value, Buffer* conv_state,
        const Buffer* grouped_qkv, const Buffer* weight, DType dtype,
        int64_t token_count, int64_t num_qk_heads,
        int64_t num_value_heads, int64_t key_head_dim,
        int64_t value_head_dim, int64_t kernel_size,
        Stream* stream = nullptr) override;
    Status deltanet_recurrent(
        Buffer* dst, Buffer* recurrent_state, const Buffer* query,
        const Buffer* key, const Buffer* value, const Buffer* a,
        const Buffer* beta, const Buffer* a_log, const Buffer* dt_bias,
        const Buffer* norm_weight, const Buffer* z, DType dtype,
        int64_t token_count, int64_t num_qk_heads,
        int64_t num_value_heads, int64_t key_head_dim,
        int64_t value_head_dim, float eps,
        Stream* stream = nullptr) override;
    Status argmax_last_row(Buffer* dst, const Buffer* src, DType dtype,
                           int64_t rows, int64_t columns,
                           Stream* stream) override;
    Status argmax_rows(Buffer* dst, const Buffer* src, DType dtype,
                       int64_t rows, int64_t columns,
                       Stream* stream = nullptr) override;

    void register_kernels() override;

private:
    struct Impl;

    Device device_;
    std::unique_ptr<Impl> impl_;
    bool kernels_registered_ = false;
};

} // namespace hybridai
