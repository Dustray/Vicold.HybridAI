#pragma once

#include "core/status.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hybridai::models {

enum class Qwen3LayerType {
    LinearAttention,
    FullAttention,
};

struct Qwen3Config {
    int64_t hidden_size = 5120;
    int64_t num_hidden_layers = 64;
    int64_t num_attention_heads = 24;
    int64_t num_key_value_heads = 4;
    int64_t head_dim = 256;
    int64_t rope_head_dim = 64;
    int64_t intermediate_size = 17408;
    int64_t vocab_size = 248320;
    int64_t max_position_embeddings = 262144;
    float rope_theta = 10000000.0f;
    float rms_norm_eps = 1.0e-6f;
    int64_t linear_num_key_heads = 16;
    int64_t linear_num_value_heads = 48;
    int64_t linear_key_head_dim = 128;
    int64_t linear_value_head_dim = 128;
    int64_t linear_conv_kernel_dim = 4;
    std::vector<Qwen3LayerType> layer_types;
    int64_t full_attention_interval = 4;
    bool attn_output_gate = true;
    std::string output_gate_type = "swish";
    std::string mamba_ssm_dtype = "float32";
    bool use_cache = true;
    int64_t bos_token_id = -1;
    std::vector<int64_t> eos_token_ids;
    int64_t pad_token_id = -1;
    bool tie_word_embeddings = false;
    int32_t mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings = false;

    Status load_json(const std::string& path);
};

} // namespace hybridai::models
