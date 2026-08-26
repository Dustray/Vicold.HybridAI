#include "models/model_weight_schema.h"

#include <charconv>

namespace hybridai::models {

Status Qwen3WeightSchema::resolve(std::string_view tensor_name,
                                  WeightKey* key) const {
    if (key == nullptr || tensor_name.empty()) {
        return Status(StatusCode::InvalidArgument,
                      "Tensor name and output key are required");
    }
    *key = WeightKey{std::string(tensor_name), -1, false};

    constexpr std::string_view prefix = "model.language_model.layers.";
    if (tensor_name.substr(0, prefix.size()) != prefix) {
        return Status::OK();
    }
    const size_t begin = prefix.size();
    const size_t end = tensor_name.find('.', begin);
    if (end == std::string_view::npos || end == begin) {
        return Status(StatusCode::InvalidModel,
                      "Invalid Qwen layer tensor name: " +
                          std::string(tensor_name));
    }
    int64_t layer = -1;
    const auto result = std::from_chars(tensor_name.data() + begin,
                                        tensor_name.data() + end, layer);
    if (result.ec != std::errc() || result.ptr != tensor_name.data() + end) {
        return Status(StatusCode::InvalidModel,
                      "Invalid Qwen layer index: " + std::string(tensor_name));
    }
    key->layer_index = layer;
    constexpr std::string_view scale_suffix = ".weight_scale_inv";
    key->is_scale = tensor_name.size() >= scale_suffix.size() &&
                    tensor_name.substr(tensor_name.size() -
                                       scale_suffix.size()) == scale_suffix;
    return Status::OK();
}

} // namespace hybridai::models