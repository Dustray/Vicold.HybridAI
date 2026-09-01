#pragma once

#include "core/dtype.h"
#include "core/status.h"

#include <cstdint>

namespace hybridai::hip_fused {

// Fuses per-head RMSNorm (with optional +1 weight offset) and partial RoPE for
// one Q tensor and one K tensor. The tensors use contiguous
// [tokens, heads, head_dim] storage. stream_handle is a native hipStream_t
// passed through the backend boundary as void*.
Status qk_rmsnorm_rope(
    void* query_dst, const void* query_src, const void* query_weight,
    void* key_dst, const void* key_src, const void* key_weight, DType dtype,
    int64_t seq_len, int64_t num_query_heads, int64_t num_kv_heads,
    int64_t head_dim, int64_t rope_head_dim, int64_t position_offset,
    float rope_base, float eps, bool add_unit_offset, void* stream_handle);

Status q_gate_rmsnorm_rope(
    void* query_dst, void* gate_dst, const void* source,
    const void* query_weight, DType dtype, int64_t seq_len,
    int64_t num_query_heads, int64_t head_dim, int64_t rope_head_dim,
    int64_t position_offset, float rope_base, float eps,
    bool add_unit_offset, void* stream_handle);

} // namespace hybridai::hip_fused
