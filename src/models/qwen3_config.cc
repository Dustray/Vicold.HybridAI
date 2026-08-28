#include "models/qwen3_config.h"

#include "core/platform.h"

#include <fstream>
#include <nlohmann/json.hpp>

namespace hybridai::models {

Status Qwen3Config::load_json(const std::string& path) {
    if (!platform::file_exists(path)) {
        return Status(StatusCode::FileNotFound,
                      "Qwen3 config does not exist: " + path);
    }

    std::ifstream file(path);
    if (!file) {
        return Status(StatusCode::FileNotFound,
                      "Failed to open Qwen3 config: " + path);
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::exception& error) {
        return Status(StatusCode::InvalidModel,
                      std::string("Invalid Qwen3 config JSON: ") + error.what());
    }

    // Qwen configs may place language-model fields under text_config.
    const nlohmann::json* config = &root;
    if (root.contains("text_config") && root.at("text_config").is_object()) {
        config = &root.at("text_config");
    }

    auto read_i64 = [&](const char* key, int64_t* target) {
        if (config->contains(key) && (*config)[key].is_number_integer()) {
            *target = (*config)[key].get<int64_t>();
        }
    };
    auto read_float = [&](const char* key, float* target) {
        if (config->contains(key) && (*config)[key].is_number()) {
            *target = (*config)[key].get<float>();
        }
    };

    read_i64("hidden_size", &hidden_size);
    read_i64("num_hidden_layers", &num_hidden_layers);
    read_i64("num_attention_heads", &num_attention_heads);
    read_i64("num_key_value_heads", &num_key_value_heads);
    read_i64("head_dim", &head_dim);
    read_i64("intermediate_size", &intermediate_size);
    read_i64("vocab_size", &vocab_size);
    read_i64("max_position_embeddings", &max_position_embeddings);
    read_float("rope_theta", &rope_theta);
    read_float("rms_norm_eps", &rms_norm_eps);
    read_i64("linear_num_key_heads", &linear_num_key_heads);
    read_i64("linear_num_value_heads", &linear_num_value_heads);
    read_i64("linear_key_head_dim", &linear_key_head_dim);
    read_i64("linear_value_head_dim", &linear_value_head_dim);
    read_i64("linear_conv_kernel_dim", &linear_conv_kernel_dim);

    // Some Qwen releases omit head_dim and derive it from hidden size and
    // attention heads. The actual Qwen3.8 config also stores rope_theta only
    // inside rope_parameters.
    if (!config->contains("head_dim") && num_attention_heads > 0) {
        head_dim = hidden_size / num_attention_heads;
    }

    if (config->contains("rope_parameters") &&
        (*config)["rope_parameters"].is_object()) {
        const auto& rope = (*config)["rope_parameters"];
        if (rope.contains("rope_theta") && rope["rope_theta"].is_number()) {
            rope_theta = rope["rope_theta"].get<float>();
        }
        if (rope.contains("partial_rotary_factor") &&
            rope["partial_rotary_factor"].is_number()) {
            rope_head_dim = static_cast<int64_t>(
                static_cast<float>(head_dim) *
                rope["partial_rotary_factor"].get<float>());
        }
    }

    if (hidden_size <= 0 || num_hidden_layers <= 0 ||
        num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0 ||
        rope_head_dim <= 0 || intermediate_size <= 0 || vocab_size <= 0 ||
        max_position_embeddings <= 0 || rope_theta <= 0.0f ||
        rms_norm_eps <= 0.0f || linear_num_key_heads <= 0 ||
        linear_num_value_heads <= 0 || linear_key_head_dim <= 0 ||
        linear_value_head_dim <= 0 || linear_conv_kernel_dim <= 0 ||
        linear_num_value_heads % linear_num_key_heads != 0 ||
        num_attention_heads % num_key_value_heads != 0 ||
        rope_head_dim > head_dim || rope_head_dim % 2 != 0) {
        return Status(StatusCode::InvalidModel,
                      "Invalid dimensions in Qwen3 config");
    }
    return Status::OK();
}

} // namespace hybridai::models
