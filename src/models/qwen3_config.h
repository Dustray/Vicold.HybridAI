#pragma once

#include "core/status.h"

#include <cstdint>
#include <string>

namespace hybridai::models {

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

    Status load_json(const std::string& path);
};

} // namespace hybridai::models
