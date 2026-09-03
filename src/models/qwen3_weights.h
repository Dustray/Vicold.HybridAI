#pragma once

#include "core/status.h"
#include "core/tensor.h"
#include "io/safetensor_loader.h"
#include "models/qwen3_config.h"
#include "models/model_weight_schema.h"
#include "models/weight_sharding.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hybridai::models {

struct Qwen3LayerWeights {
    int64_t layer_index = -1;
    bool is_attention_layer = false;

    Tensor input_layernorm;
    Tensor post_attention_layernorm;
    Tensor linear_attn_norm;
    Tensor q_norm;
    Tensor k_norm;

    Tensor a_log;
    Tensor conv1d_weight;
    Tensor dt_bias;
    Tensor in_proj_a;
    Tensor in_proj_b;
    ops::QuantizedWeight in_proj_qkv;
    ops::QuantizedWeight in_proj_z;
    ops::QuantizedWeight linear_out_proj;

    ops::QuantizedWeight q_proj;
    ops::QuantizedWeight k_proj;
    ops::QuantizedWeight v_proj;
    ops::QuantizedWeight o_proj;

    ops::QuantizedWeight mlp_gate_proj;
    ops::QuantizedWeight mlp_up_proj;
    ops::QuantizedWeight mlp_down_proj;

    Status validate() const;
};

struct Qwen3SharedWeights {
    Tensor embed_tokens;
    Tensor final_norm;
    Tensor lm_head;
};

struct Qwen3MTPWeights {
    Tensor fc;
    Tensor norm;
    Tensor pre_fc_norm_embedding;
    Tensor pre_fc_norm_hidden;
    std::vector<Qwen3LayerWeights> layers;

    Status validate() const;
};

struct Qwen3DevicePartition {
    Device device;
    int64_t first_layer = 0;
    int64_t last_layer = -1;
    uint64_t weight_bytes = 0;
};

// Fully resident text-model weights. Layers are stored by model index while
// layer_devices records the device that owns each layer. Embeddings live on
// the first device; final norm and LM head live on the last device.
struct Qwen3DistributedWeights {
    Qwen3SharedWeights shared;
    Qwen3MTPWeights mtp;
    std::vector<Qwen3LayerWeights> layers;
    std::vector<Device> layer_devices;
    std::vector<Qwen3DevicePartition> partitions;
    uint64_t total_weight_bytes = 0;
};

struct Qwen3LoadOptions {
    // Phase 1 is text-only. Vision is opt-in and remains disabled by default.
    bool enable_vision = false;
};

// Qwen-specific assembly layer. SafeTensorLoader remains model agnostic; this
// class owns Qwen names, layer structure, scale pairing, and lazy loading.
class Qwen3WeightLoader {
public:
    Status open(const std::string& model_path);
    Status open(const std::string& model_path,
                const Qwen3LoadOptions& options);
    Status open(const std::string& model_path, const Qwen3Config& config);
    Status open(const std::string& model_path, const Qwen3Config& config,
                const Qwen3LoadOptions& options);

    bool is_open() const noexcept { return loader_ != nullptr; }
    const Qwen3Config& config() const noexcept { return config_; }
    const Qwen3LoadOptions& options() const noexcept { return options_; }

    Status load_layer(int64_t layer_index, Device device,
                      Qwen3LayerWeights* output) const;
    Status load_shared(Device device, Qwen3SharedWeights* output) const;
    Status load_mtp(Device device, Qwen3MTPWeights* output) const;
    Status plan_distributed(const std::vector<Device>& devices,
                            std::vector<Qwen3DevicePartition>* partitions) const;
    Status load_distributed(const std::vector<Device>& devices,
                            Qwen3DistributedWeights* output) const;

private:
    Status load_tensor(const std::string& name, Device device,
                       Tensor* output) const;
    Status load_quantized(const std::string& base_name, Device device,
                          ops::QuantizedWeight* output) const;
    Status load_layer_tensor(int64_t layer_index, const std::string& suffix,
                             Device device, Tensor* output) const;
    std::string layer_name(int64_t layer_index, const std::string& suffix) const;
    uint64_t tensor_bytes(const std::string& name) const;
    uint64_t layer_bytes(int64_t layer_index) const;
    uint64_t mtp_bytes() const;

    std::shared_ptr<io::SafeTensorLoader> loader_;
    Qwen3Config config_;
    Qwen3LoadOptions options_;
    Qwen3WeightSchema schema_;
};

} // namespace hybridai::models
