#include "models/qwen3_config.h"

#include "core/platform.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

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

    // Qwen configs may place language-model fields under text_config. Resolve
    // every field independently so root-level compatibility fields remain
    // visible when a partial text_config is present.
    const nlohmann::json* text_config = nullptr;
    if (root.contains("text_config") && root.at("text_config").is_object())
        text_config = &root.at("text_config");
    auto field = [&](const char* key) -> const nlohmann::json* {
        if (text_config != nullptr && text_config->contains(key))
            return &text_config->at(key);
        if (root.contains(key)) return &root.at(key);
        return nullptr;
    };

    auto read_i64 = [&](const char* key, int64_t* target) {
        const auto* value = field(key);
        if (value != nullptr && value->is_number_integer())
            *target = value->get<int64_t>();
    };
    auto read_float = [&](const char* key, float* target) {
        const auto* value = field(key);
        if (value != nullptr && value->is_number())
            *target = value->get<float>();
    };
    auto read_bool = [&](const char* key, bool* target) {
        const auto* value = field(key);
        if (value != nullptr && value->is_boolean())
            *target = value->get<bool>();
    };
    auto read_string = [&](const char* key, std::string* target) {
        const auto* value = field(key);
        if (value != nullptr && value->is_string())
            *target = value->get<std::string>();
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
    read_i64("full_attention_interval", &full_attention_interval);
    read_bool("attn_output_gate", &attn_output_gate);
    read_string("output_gate_type", &output_gate_type);
    read_string("mamba_ssm_dtype", &mamba_ssm_dtype);
    read_bool("use_cache", &use_cache);
    read_i64("bos_token_id", &bos_token_id);
    read_i64("pad_token_id", &pad_token_id);
    read_bool("tie_word_embeddings", &tie_word_embeddings);

    eos_token_ids.clear();
    if (const auto* eos = field("eos_token_id"); eos != nullptr) {
        if (eos->is_number_integer()) {
            eos_token_ids.push_back(eos->get<int64_t>());
        } else if (eos->is_array()) {
            for (const auto& id : *eos) {
                if (!id.is_number_integer()) {
                    return Status(StatusCode::InvalidModel,
                                  "Qwen3 eos_token_id contains a non-integer value");
                }
                eos_token_ids.push_back(id.get<int64_t>());
            }
        } else if (!eos->is_null()) {
            return Status(StatusCode::InvalidModel,
                          "Qwen3 eos_token_id must be an integer or array");
        }
    }

    // Some Qwen releases omit head_dim and derive it from hidden size and
    // attention heads. The actual Qwen3.8 config also stores rope_theta only
    // inside rope_parameters.
    if (field("head_dim") == nullptr && num_attention_heads > 0) {
        head_dim = hidden_size / num_attention_heads;
    }

    const auto* rope_parameters = field("rope_parameters");
    if (rope_parameters != nullptr && rope_parameters->is_object()) {
        const auto& rope = *rope_parameters;
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

    layer_types.clear();
    const auto* schedule = field("layer_types");
    if (schedule != nullptr) {
        if (!schedule->is_array()) {
            return Status(StatusCode::InvalidModel,
                          "Qwen3 layer_types must be an array");
        }
        layer_types.reserve(schedule->size());
        for (size_t index = 0; index < schedule->size(); ++index) {
            const auto& value = schedule->at(index);
            if (!value.is_string()) {
                return Status(StatusCode::InvalidModel,
                              "Qwen3 layer_types contains a non-string value");
            }
            const std::string type = value.get<std::string>();
            if (type == "linear_attention") {
                layer_types.push_back(Qwen3LayerType::LinearAttention);
            } else if (type == "full_attention") {
                layer_types.push_back(Qwen3LayerType::FullAttention);
            } else {
                std::ostringstream message;
                message << "Unknown Qwen3 layer type at index " << index
                        << ": " << type;
                return Status(StatusCode::InvalidModel, message.str());
            }
        }
    } else if (full_attention_interval > 0) {
        layer_types.reserve(static_cast<size_t>(num_hidden_layers));
        for (int64_t index = 0; index < num_hidden_layers; ++index) {
            layer_types.push_back(
                (index + 1) % full_attention_interval == 0
                    ? Qwen3LayerType::FullAttention
                    : Qwen3LayerType::LinearAttention);
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
        rope_head_dim > head_dim || rope_head_dim % 2 != 0 ||
        full_attention_interval <= 0 ||
        layer_types.size() != static_cast<size_t>(num_hidden_layers)) {
        return Status(StatusCode::InvalidModel,
                      "Invalid dimensions in Qwen3 config");
    }
    for (int64_t index = 0; index < num_hidden_layers; ++index) {
        const Qwen3LayerType expected =
            (index + 1) % full_attention_interval == 0
                ? Qwen3LayerType::FullAttention
                : Qwen3LayerType::LinearAttention;
        if (layer_types[static_cast<size_t>(index)] != expected) {
            return Status(StatusCode::InvalidModel,
                          "Qwen3 layer_types conflicts with full_attention_interval");
        }
    }
    if (!attn_output_gate || output_gate_type != "swish" ||
        mamba_ssm_dtype != "float32") {
        return Status(StatusCode::InvalidModel,
                      "Unsupported Qwen3 attention gate or SSM state configuration");
    }
    return Status::OK();
}

} // namespace hybridai::models
