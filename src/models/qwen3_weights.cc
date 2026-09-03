#include "models/qwen3_weights.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <sstream>

namespace hybridai::models {
namespace {

Status Missing(const std::string& name) {
    return Status(StatusCode::InvalidModel, "Missing Qwen weight: " + name);
}

bool Has(const io::SafeTensorLoader& loader, const std::string& name) {
    return loader.tensor_info(name) != nullptr;
}

} // namespace

Status Qwen3LayerWeights::validate() const {
    if (layer_index < 0) {
        return Status(StatusCode::InvalidArgument, "Invalid Qwen layer index");
    }
    if (input_layernorm.buffer() == nullptr ||
        post_attention_layernorm.buffer() == nullptr) {
        return Status(StatusCode::InvalidModel,
                      "Qwen layer norms are not loaded");
    }
    if (is_attention_layer) {
        if (!q_proj.validate().ok() || !k_proj.validate().ok() ||
            !v_proj.validate().ok() || !o_proj.validate().ok()) {
            return Status(StatusCode::InvalidModel,
                          "Qwen attention projections are not loaded");
        }
    } else {
        if (!in_proj_qkv.validate().ok() || !in_proj_z.validate().ok() ||
            !linear_out_proj.validate().ok()) {
            return Status(StatusCode::InvalidModel,
                          "Qwen DeltaNet weights are incomplete");
        }
    }
    return mlp_gate_proj.validate().ok() && mlp_up_proj.validate().ok() &&
                   mlp_down_proj.validate().ok()
               ? Status::OK()
               : Status(StatusCode::InvalidModel,
                        "Qwen MLP quantized weights are invalid");
}

Status Qwen3MTPWeights::validate() const {
    if (fc.buffer() == nullptr || norm.buffer() == nullptr ||
        pre_fc_norm_embedding.buffer() == nullptr ||
        pre_fc_norm_hidden.buffer() == nullptr) {
        return Status(StatusCode::InvalidModel,
                      "Qwen MTP projection or normalization weights are missing");
    }
    if (fc.shape().ndim() != 2 || norm.shape().ndim() != 1 ||
        pre_fc_norm_embedding.shape().ndim() != 1 ||
        pre_fc_norm_hidden.shape().ndim() != 1 ||
        norm.shape() != pre_fc_norm_embedding.shape() ||
        norm.shape() != pre_fc_norm_hidden.shape() ||
        fc.shape().dim(1) != norm.shape().dim(0) * 2 ||
        fc.shape().dim(0) != norm.shape().dim(0)) {
        return Status(StatusCode::InvalidModel,
                      "Qwen MTP projection or normalization shapes are invalid");
    }
    if (layers.empty()) {
        return Status(StatusCode::InvalidModel,
                      "Qwen MTP decoder layers are missing");
    }
    for (const auto& layer : layers) {
        if (layer.q_norm.buffer() == nullptr || layer.k_norm.buffer() == nullptr) {
            return Status(StatusCode::InvalidModel,
                          "Qwen MTP attention normalization weights are missing");
        }
        Status status = layer.validate();
        if (!status.ok()) return status;
    }
    return Status::OK();
}

Status Qwen3WeightLoader::open(const std::string& model_path) {
    return open(model_path, Qwen3LoadOptions{});
}

Status Qwen3WeightLoader::open(const std::string& model_path,
                               const Qwen3LoadOptions& options) {
    Qwen3Config config;
    const auto config_path = std::filesystem::path(model_path) / "config.json";
    Status status = config.load_json(config_path.string());
    if (!status.ok()) return status;
    return open(model_path, config, options);
}

Status Qwen3WeightLoader::open(const std::string& model_path,
                               const Qwen3Config& config) {
    return open(model_path, config, Qwen3LoadOptions{});
}

Status Qwen3WeightLoader::open(const std::string& model_path,
                               const Qwen3Config& config,
                               const Qwen3LoadOptions& options) {
    if (options.enable_vision) {
        return Status(StatusCode::NotImplemented,
                      "Qwen vision loading is disabled in phase 1");
    }
    auto loader = std::make_shared<io::SafeTensorLoader>();
    Status status = loader->open_model(model_path);
    if (!status.ok()) return status;
    loader_ = std::move(loader);
    config_ = config;
    options_ = options;
    return Status::OK();
}

namespace {
MemoryType MemoryTypeForDevice(Device device) {
    return device.is_cpu() ? MemoryType::Host : MemoryType::Device;
}
} // namespace

Status Qwen3WeightLoader::load_tensor(const std::string& name, Device device,
                                      Tensor* output) const {
    if (!loader_) {
        return Status(StatusCode::InvalidArgument, "Qwen loader is not open");
    }
    if (!Has(*loader_, name)) return Missing(name);
    return loader_->load(name, device, output, MemoryTypeForDevice(device));
}

Status Qwen3WeightLoader::load_quantized(
    const std::string& base_name, Device device,
    ops::QuantizedWeight* output) const {
    if (output == nullptr) {
        return Status(StatusCode::InvalidArgument, "Output is null");
    }
    constexpr std::string_view weight_suffix = ".weight";
    if (base_name.size() < weight_suffix.size() ||
        base_name.compare(base_name.size() - weight_suffix.size(),
                          weight_suffix.size(), weight_suffix) != 0) {
        return Status(StatusCode::InvalidArgument,
                      "Quantized Qwen weight name must end with .weight");
    }
    const std::string scale_name =
        base_name.substr(0, base_name.size() - weight_suffix.size()) +
        ".weight_scale_inv";
    if (!Has(*loader_, base_name)) return Missing(base_name);
    const auto* weight_info = loader_->tensor_info(base_name);
    if (weight_info->dtype == DType::BF16 ||
        weight_info->dtype == DType::FP16 ||
        weight_info->dtype == DType::FP32) {
        Status status = loader_->load(base_name, device, &output->values,
                                      MemoryTypeForDevice(device));
        if (!status.ok()) return status;
        output->scales = Tensor();
        output->quantization = ops::QuantizationSpec{};
        return output->validate();
    }
    if (weight_info->dtype != DType::FP8_E4M3 ||
        !Has(*loader_, scale_name)) {
        return Status(StatusCode::UnsupportedDType,
                      "Expected dense BF16/FP16/FP32 or scaled FP8 Qwen weight: " +
                          base_name);
    }

    Tensor values;
    Status status = loader_->load(base_name, device, &values,
                                  MemoryTypeForDevice(device));
    if (!status.ok()) return status;
    Tensor scales;
    // Scales are small and currently consumed on the host (until a HIP
    // dequant kernel is wired up), so keep them in host memory.
    status = loader_->load_as_fp32(scale_name, device, &scales,
                                    MemoryType::Host);
    if (!status.ok()) return status;

    output->values = std::move(values);
    output->scales = std::move(scales);
    output->quantization = ops::QuantizationSpec{
        ops::ScaleLayout::Block2D, 128, 128};
    return output->validate();
}

std::string Qwen3WeightLoader::layer_name(int64_t layer_index,
                                          const std::string& suffix) const {
    return "model.language_model.layers." + std::to_string(layer_index) +
           "." + suffix;
}

Status Qwen3WeightLoader::load_layer_tensor(int64_t layer_index,
                                            const std::string& suffix,
                                            Device device, Tensor* output) const {
    return load_tensor(layer_name(layer_index, suffix), device, output);
}

Status Qwen3WeightLoader::load_layer(int64_t layer_index, Device device,
                                     Qwen3LayerWeights* output) const {
    if (output == nullptr || layer_index < 0 ||
        layer_index >= config_.num_hidden_layers) {
        return Status(StatusCode::InvalidArgument, "Invalid Qwen layer request");
    }
    if (config_.layer_types.size() !=
        static_cast<size_t>(config_.num_hidden_layers)) {
        return Status(StatusCode::InvalidModel,
                      "Qwen layer schedule is missing or incomplete");
    }
    Qwen3LayerWeights result;
    result.layer_index = layer_index;
    Status status = load_layer_tensor(layer_index, "input_layernorm.weight",
                                      device, &result.input_layernorm);
    if (!status.ok()) return status;
    status = load_layer_tensor(layer_index, "post_attention_layernorm.weight",
                               device, &result.post_attention_layernorm);
    if (!status.ok()) return status;

    const bool has_attention = Has(
        *loader_, layer_name(layer_index, "self_attn.q_proj.weight"));
    const bool has_linear = Has(
        *loader_, layer_name(layer_index, "linear_attn.in_proj_qkv.weight"));
    if (has_attention == has_linear) {
        return Status(
            StatusCode::InvalidModel,
            "Qwen layer " + std::to_string(layer_index) +
                (has_attention
                     ? " contains both full-attention and linear-attention weights"
                     : " contains neither full-attention nor linear-attention weights"));
    }
    const Qwen3LayerType expected =
        config_.layer_types[static_cast<size_t>(layer_index)];
    result.is_attention_layer = expected == Qwen3LayerType::FullAttention;
    if (result.is_attention_layer != has_attention) {
        return Status(
            StatusCode::InvalidModel,
            "Qwen layer " + std::to_string(layer_index) +
                (result.is_attention_layer
                     ? " is configured as full attention but checkpoint contains linear attention"
                     : " is configured as linear attention but checkpoint contains full attention"));
    }
    if (result.is_attention_layer) {
        for (const char* suffix : {"self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                   "self_attn.v_proj.weight", "self_attn.o_proj.weight"}) {
            ops::QuantizedWeight* target = nullptr;
            if (std::string(suffix).find("q_proj") != std::string::npos) target = &result.q_proj;
            else if (std::string(suffix).find("k_proj") != std::string::npos) target = &result.k_proj;
            else if (std::string(suffix).find("v_proj") != std::string::npos) target = &result.v_proj;
            else target = &result.o_proj;
            status = load_quantized(layer_name(layer_index, suffix), device, target);
            if (!status.ok()) return status;
        }
        status = load_layer_tensor(layer_index, "self_attn.q_norm.weight",
                                   device, &result.q_norm);
        if (!status.ok()) return status;
        status = load_layer_tensor(layer_index, "self_attn.k_norm.weight",
                                   device, &result.k_norm);
        if (!status.ok()) return status;
    } else {
        for (const auto& item : {std::pair{"linear_attn.norm.weight", &result.linear_attn_norm},
                                 std::pair{"linear_attn.A_log", &result.a_log},
                                 std::pair{"linear_attn.conv1d.weight", &result.conv1d_weight},
                                 std::pair{"linear_attn.dt_bias", &result.dt_bias},
                                 std::pair{"linear_attn.in_proj_a.weight", &result.in_proj_a},
                                 std::pair{"linear_attn.in_proj_b.weight", &result.in_proj_b}}) {
            status = load_layer_tensor(layer_index, item.first, device, item.second);
            if (!status.ok()) return status;
        }
        status = load_quantized(layer_name(layer_index, "linear_attn.in_proj_qkv.weight"), device,
                                &result.in_proj_qkv);
        if (!status.ok()) return status;
        status = load_quantized(layer_name(layer_index, "linear_attn.in_proj_z.weight"), device,
                                &result.in_proj_z);
        if (!status.ok()) return status;
        status = load_quantized(layer_name(layer_index, "linear_attn.out_proj.weight"), device,
                                &result.linear_out_proj);
        if (!status.ok()) return status;
    }

    status = load_quantized(layer_name(layer_index, "mlp.gate_proj.weight"), device,
                            &result.mlp_gate_proj);
    if (!status.ok()) return status;
    status = load_quantized(layer_name(layer_index, "mlp.up_proj.weight"), device,
                            &result.mlp_up_proj);
    if (!status.ok()) return status;
    status = load_quantized(layer_name(layer_index, "mlp.down_proj.weight"), device,
                            &result.mlp_down_proj);
    if (!status.ok()) return status;

    status = result.validate();
    if (!status.ok()) return status;
    *output = std::move(result);
    return Status::OK();
}

Status Qwen3WeightLoader::load_shared(Device device,
                                      Qwen3SharedWeights* output) const {
    if (output == nullptr) {
        return Status(StatusCode::InvalidArgument, "Output is null");
    }
    Qwen3SharedWeights result;
    Status status = load_tensor("model.language_model.embed_tokens.weight", device,
                                &result.embed_tokens);
    if (!status.ok()) return status;
    status = load_tensor("model.language_model.norm.weight", device,
                         &result.final_norm);
    if (!status.ok()) return status;
    status = load_tensor("lm_head.weight", device, &result.lm_head);
    if (!status.ok()) return status;
    *output = std::move(result);
    return Status::OK();
}

Status Qwen3WeightLoader::load_mtp(Device device,
                                   Qwen3MTPWeights* output) const {
    if (output == nullptr) {
        return Status(StatusCode::InvalidArgument, "Output is null");
    }
    if (config_.mtp_num_hidden_layers <= 0) {
        return Status(StatusCode::InvalidModel,
                      "Checkpoint does not contain Qwen MTP layers");
    }

    Qwen3MTPWeights result;
    Status status = load_tensor("mtp.fc.weight", device, &result.fc);
    if (!status.ok()) return status;
    status = load_tensor("mtp.norm.weight", device, &result.norm);
    if (!status.ok()) return status;
    status = load_tensor("mtp.pre_fc_norm_embedding.weight", device,
                         &result.pre_fc_norm_embedding);
    if (!status.ok()) return status;
    status = load_tensor("mtp.pre_fc_norm_hidden.weight", device,
                         &result.pre_fc_norm_hidden);
    if (!status.ok()) return status;

    result.layers.resize(static_cast<size_t>(config_.mtp_num_hidden_layers));
    for (int32_t index = 0; index < config_.mtp_num_hidden_layers; ++index) {
        const std::string prefix = "mtp.layers." + std::to_string(index) + ".";
        Qwen3LayerWeights layer;
        layer.layer_index = index;
        layer.is_attention_layer = true;
        auto load = [&](const std::string& suffix, Tensor* tensor) {
            return load_tensor(prefix + suffix, device, tensor);
        };
        status = load("input_layernorm.weight", &layer.input_layernorm);
        if (!status.ok()) return status;
        status = load("post_attention_layernorm.weight",
                      &layer.post_attention_layernorm);
        if (!status.ok()) return status;
        status = load_quantized(prefix + "self_attn.q_proj.weight", device,
                                &layer.q_proj);
        if (!status.ok()) return status;
        status = load_quantized(prefix + "self_attn.k_proj.weight", device,
                                &layer.k_proj);
        if (!status.ok()) return status;
        status = load_quantized(prefix + "self_attn.v_proj.weight", device,
                                &layer.v_proj);
        if (!status.ok()) return status;
        status = load_quantized(prefix + "self_attn.o_proj.weight", device,
                                &layer.o_proj);
        if (!status.ok()) return status;
        status = load("self_attn.q_norm.weight", &layer.q_norm);
        if (!status.ok()) return status;
        status = load("self_attn.k_norm.weight", &layer.k_norm);
        if (!status.ok()) return status;
        status = load_quantized(prefix + "mlp.gate_proj.weight", device,
                                &layer.mlp_gate_proj);
        if (!status.ok()) return status;
        status = load_quantized(prefix + "mlp.up_proj.weight", device,
                                &layer.mlp_up_proj);
        if (!status.ok()) return status;
        status = load_quantized(prefix + "mlp.down_proj.weight", device,
                                &layer.mlp_down_proj);
        if (!status.ok()) return status;
        status = layer.validate();
        if (!status.ok()) return status;
        result.layers[static_cast<size_t>(index)] = std::move(layer);
    }
    status = result.validate();
    if (!status.ok()) return status;
    *output = std::move(result);
    return Status::OK();
}

uint64_t Qwen3WeightLoader::mtp_bytes() const {
    if (!loader_ || config_.mtp_num_hidden_layers <= 0) return 0;
    uint64_t bytes = 0;
    for (const std::string& name : loader_->tensor_names()) {
        if (name == "mtp.fc.weight" || name == "mtp.norm.weight" ||
            name == "mtp.pre_fc_norm_embedding.weight" ||
            name == "mtp.pre_fc_norm_hidden.weight" ||
            name.compare(0, std::string("mtp.layers.").size(), "mtp.layers.") == 0) {
            bytes += tensor_bytes(name);
        }
    }
    return bytes;
}

Status Qwen3WeightLoader::plan_distributed(
    const std::vector<Device>& devices,
    std::vector<Qwen3DevicePartition>* partitions) const {
    if (!loader_) {
        return Status(StatusCode::InvalidArgument, "Qwen loader is not open");
    }
    if (partitions == nullptr || devices.empty()) {
        return Status(StatusCode::InvalidArgument,
                      "Distributed plan requires output and at least one device");
    }
    for (const Device& device : devices) {
        if (!device.is_gpu()) {
            return Status(StatusCode::InvalidDevice,
                          "Distributed Qwen plan requires GPU devices");
        }
    }

    const int64_t layer_count = config_.num_hidden_layers;
    const int64_t device_count = static_cast<int64_t>(devices.size());
    if (layer_count <= 0 || device_count > layer_count) {
        return Status(StatusCode::InvalidArgument,
                      "Qwen device count must not exceed layer count");
    }

    std::vector<Qwen3DevicePartition> result(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        result[i].device = devices[i];
        result[i].first_layer = layer_count * static_cast<int64_t>(i) /
                                device_count;
        result[i].last_layer = layer_count * static_cast<int64_t>(i + 1) /
                                   device_count - 1;
        for (int64_t layer = result[i].first_layer;
             layer <= result[i].last_layer; ++layer) {
            result[i].weight_bytes += layer_bytes(layer);
        }
    }
    result.front().weight_bytes +=
        tensor_bytes("model.language_model.embed_tokens.weight");
    result.back().weight_bytes +=
        tensor_bytes("model.language_model.norm.weight");
    result.back().weight_bytes += tensor_bytes("lm_head.weight");
    result.back().weight_bytes += mtp_bytes();
    *partitions = std::move(result);
    return Status::OK();
}

uint64_t Qwen3WeightLoader::tensor_bytes(const std::string& name) const {
    if (!loader_) return 0;
    const auto* info = loader_->tensor_info(name);
    return info == nullptr ? 0 : info->data_end - info->data_begin;
}

uint64_t Qwen3WeightLoader::layer_bytes(int64_t layer_index) const {
    if (!loader_) return 0;
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer_index) + ".";
    uint64_t bytes = 0;
    for (const std::string& name : loader_->tensor_names()) {
        if (name.compare(0, prefix.size(), prefix) == 0) {
            bytes += tensor_bytes(name);
        }
    }
    return bytes;
}

Status Qwen3WeightLoader::load_distributed(
    const std::vector<Device>& devices, Qwen3DistributedWeights* output) const {
    if (!loader_) {
        return Status(StatusCode::InvalidArgument, "Qwen loader is not open");
    }
    if (output == nullptr || devices.empty()) {
        return Status(StatusCode::InvalidArgument,
                      "Distributed load requires output and at least one device");
    }

    Qwen3DistributedWeights result;
    result.layers.resize(static_cast<size_t>(config_.num_hidden_layers));
    result.layer_devices.resize(static_cast<size_t>(config_.num_hidden_layers));
    Status status = plan_distributed(devices, &result.partitions);
    if (!status.ok()) return status;

    for (const auto& partition : result.partitions) {
        for (int64_t layer_index = partition.first_layer;
             layer_index <= partition.last_layer; ++layer_index) {
            status = load_layer(layer_index, partition.device,
                                &result.layers[static_cast<size_t>(layer_index)]);
            if (!status.ok()) return status;
            result.layer_devices[static_cast<size_t>(layer_index)] =
                partition.device;
        }
        result.total_weight_bytes += partition.weight_bytes;
    }

    status = load_tensor("model.language_model.embed_tokens.weight",
                         devices.front(), &result.shared.embed_tokens);
    if (!status.ok()) return status;
    status = load_tensor("model.language_model.norm.weight", devices.back(),
                         &result.shared.final_norm);
    if (!status.ok()) return status;
    status = load_tensor("lm_head.weight", devices.back(),
                         &result.shared.lm_head);
    if (!status.ok()) return status;

    *output = std::move(result);
    return Status::OK();
}

} // namespace hybridai::models
