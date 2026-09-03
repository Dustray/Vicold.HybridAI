#pragma once

#include "core/device.h"
#include "core/status.h"
#include "core/tensor.h"
#include "models/qwen3_config.h"
#include "models/qwen3_weights.h"
#include "ops/attention.h"
#include "ops/delta_net.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hybridai {

namespace ops {
struct AttentionKVCache;
struct DeltaNetCache;
}

std::vector<Device> select_gpu_devices(const std::string& backend_name,
                                       int max_devices);

Status lookup_embedding(const Tensor& embedding, const std::vector<int64_t>& ids,
                        Backend* backend, const Device& device,
                        Tensor* output);

Status lookup_embedding_into(const Tensor& embedding,
                             const std::vector<int64_t>& ids,
                             Backend* backend, const Device& device,
                             const std::shared_ptr<Buffer>& output_buffer,
                             const std::shared_ptr<Buffer>& ids_buffer,
                             Tensor* output);

Tensor qwen_rmsnorm_reference(const Tensor& input, const Tensor& weight,
                              Backend* backend, const Device& device,
                              float eps = 1e-6f);

Tensor make_residual(const Tensor& lhs, const Tensor& rhs, Backend* backend,
                     const Device& device);

Tensor qwen_mlp_reference(const Tensor& input,
                          const models::Qwen3LayerWeights& layer,
                          Backend* backend, const Device& device);

Tensor qwen_attention_reference(
    const Tensor& input, const models::Qwen3LayerWeights& layer,
    Backend* backend, const Device& device,
    const models::Qwen3Config& config, ops::AttentionKVCache* cache = nullptr,
    int64_t max_cache_len = 0);

Tensor qwen_deltanet_reference(
    const Tensor& input, const models::Qwen3LayerWeights& layer,
    Backend* backend, const Device& device,
    const models::Qwen3Config& config, ops::DeltaNetCache* cache = nullptr);

}  // namespace hybridai
